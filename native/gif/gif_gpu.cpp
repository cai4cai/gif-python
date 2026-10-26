// SPDX-License-Identifier: BSD-3-Clause
// gif_gpu - GPU backend of gif-python's multi-atlas segmentation (src/multi_atlas/multi_atlas_segmentation.py).
//
//   gif_gpu --img target.nii.gz --out OUT --plan plan.txt --atlas NAME img.nii.gz seg.nii.gz [--atlas ...] [options]
//
// Computes what gif-python computes with reg_aladin, reg_f3d, reg_resample, numpy/scipy, numba and seg_EM, on the GPU, and writes the
// same output files (multi_atlas_proba_seg_argmax / tissue_prior / tissue_seg / final_parcellation .nii.gz and, optionally,
// registered_atlases/<name>/warped_atlas_seg.nii.gz).  Run with --help for the options.  Written from gif-python's Python code.
#include <cuda_runtime.h>
#include <omp.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <atomic>
#include <future>
#include <thread>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <vector>

#include "../gpu/aladin/aladin.h"
#include "../gpu/common/gzwrite.h"
#include "../gpu/f3d/include/f3d.h"
#include "../gpu/seg/include/segem.h"
#include "gif_io.h"
#include "gif_stages.h"

using namespace f3d;
using Clock = std::chrono::steady_clock;

namespace {

struct Atlas {
    std::string name, img, seg;
};

struct Cfg {
    std::string img, out, plan, structure_info = "structure info csv";
    std::vector<Atlas> atlases;
    // reg_aladin
    int aladin_lp = 2;
    // reg_f3d
    float f3d_jl = 0.0001f, f3d_be = 0.005f, f3d_sx = -5.f, f3d_lncc = 5.f;
    int f3d_maxit = 250, f3d_ln = 4, f3d_lp = 3, f3d_interp = 1;
    // weights
    float lncc_sigma[3] = {-2.5f, -2.5f, -2.5f};
    float temperature = 0.15f;
    // seg_EM
    int em_maxit = 30, em_minit = 3, em_bias_order = 4;
    float em_bias_thresh = 0.05f, em_mrf_beta = 0.1f;
    // execution
    int numerics = 0;
    int f3d_batch = 20;
    int write_warped_seg = 1;
    int gz_level = 1;
    int gz_threads = 16;
    std::string cpp_dir, affine_dir, dump_dir, timing_json;
    bool verbose = false;
};

[[noreturn]] void fail(int code, const std::string &msg)
{
    fprintf(stderr, "gif_gpu: %s\n", msg.c_str());
    exit(code);
}

void usage()
{
    printf(
        "usage: gif_gpu --img IMG --out DIR --plan FILE --atlas NAME IMG SEG [--atlas ...] [options]\n"
        "  --plan FILE            'tissues T' and one 'structure LABEL N t1 .. tN' line per label of the structure info csv\n"
        "  --aladin-lp N          reg_aladin -lp (2)\n"
        "  --f3d-jl X --f3d-be X --f3d-maxit N --f3d-ln N --f3d-lp N --f3d-sx X --f3d-lncc X --f3d-interp 1\n"
        "                         reg_f3d -jl -be -maxit -ln -lp -sx --lncc --interp (0.0001 0.005 250 4 3 -5 5 1)\n"
        "  --lncc-sigma X Y Z     LNCC kernel of the fusion weights (-2.5 -2.5 -2.5; negative: mm, positive: voxels)\n"
        "  --temperature X        weights = exp(-lncc distance / X) (0.15)\n"
        "  --em-maxit N --em-minit N --em-bias-order N --em-bias-thresh X --em-mrf-beta X   seg_EM (30 3 4 0.05 0.1)\n"
        "  --numerics 0|1         F3D numerics: 0 exact (default), 1 validated fast option\n"
        "  --f3d-batch N          atlases registered together by F3D (20); lower it to save GPU memory\n"
        "  --write-warped-seg 0|1 write registered_atlases/NAME/warped_atlas_seg.nii.gz (1)\n"
        "  --cpp-dir DIR          test mode: skip the registration and use DIR/NAME/cpp.nii.gz (control point grids of a CPU run)\n"
        "  --affine-dir DIR       test mode: use DIR/NAME/affine.txt (reg_aladin result of a CPU run) instead of the GPU reg_aladin\n"
        "  --dump-dir DIR         write affine.txt, cpp.nii.gz, weights.nii.gz and warped_atlas_img_linear_interp.nii.gz per atlas\n"
        "  --timing-json FILE     write per-stage timings\n"
        "  --check-device         print the CUDA device and exit (status 5 if there is none)\n");
}

Cfg parse(int argc, char **argv)
{
    Cfg c;
    auto need = [&](int i, int n) {
        if (i + n >= argc) fail(2, std::string("missing value for ") + argv[i]);
    };
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--help" || a == "-h") { usage(); exit(0); }
        else if (a == "--img") { need(i, 1); c.img = argv[++i]; }
        else if (a == "--out") { need(i, 1); c.out = argv[++i]; }
        else if (a == "--plan") { need(i, 1); c.plan = argv[++i]; }
        else if (a == "--structure-info") { need(i, 1); c.structure_info = argv[++i]; }
        else if (a == "--atlas") { need(i, 3); Atlas t; t.name = argv[++i]; t.img = argv[++i]; t.seg = argv[++i]; c.atlases.push_back(t); }
        else if (a == "--aladin-lp") { need(i, 1); c.aladin_lp = atoi(argv[++i]); }
        else if (a == "--f3d-jl") { need(i, 1); c.f3d_jl = (float)atof(argv[++i]); }
        else if (a == "--f3d-be") { need(i, 1); c.f3d_be = (float)atof(argv[++i]); }
        else if (a == "--f3d-maxit") { need(i, 1); c.f3d_maxit = atoi(argv[++i]); }
        else if (a == "--f3d-ln") { need(i, 1); c.f3d_ln = atoi(argv[++i]); }
        else if (a == "--f3d-lp") { need(i, 1); c.f3d_lp = atoi(argv[++i]); }
        else if (a == "--f3d-sx") { need(i, 1); c.f3d_sx = (float)atof(argv[++i]); }
        else if (a == "--f3d-lncc") { need(i, 1); c.f3d_lncc = (float)atof(argv[++i]); }
        else if (a == "--f3d-interp") { need(i, 1); c.f3d_interp = atoi(argv[++i]); }
        else if (a == "--lncc-sigma") { need(i, 3); for (int k = 0; k < 3; k++) c.lncc_sigma[k] = (float)atof(argv[++i]); }
        else if (a == "--temperature") { need(i, 1); c.temperature = (float)atof(argv[++i]); }
        else if (a == "--em-maxit") { need(i, 1); c.em_maxit = atoi(argv[++i]); }
        else if (a == "--em-minit") { need(i, 1); c.em_minit = atoi(argv[++i]); }
        else if (a == "--em-bias-order") { need(i, 1); c.em_bias_order = atoi(argv[++i]); }
        else if (a == "--em-bias-thresh") { need(i, 1); c.em_bias_thresh = (float)atof(argv[++i]); }
        else if (a == "--em-mrf-beta") { need(i, 1); c.em_mrf_beta = (float)atof(argv[++i]); }
        else if (a == "--numerics") { need(i, 1); c.numerics = atoi(argv[++i]); }
        else if (a == "--f3d-batch") { need(i, 1); c.f3d_batch = atoi(argv[++i]); }
        else if (a == "--write-warped-seg") { need(i, 1); c.write_warped_seg = atoi(argv[++i]); }
        else if (a == "--gz-level") { need(i, 1); c.gz_level = atoi(argv[++i]); }
        else if (a == "--gz-threads") { need(i, 1); c.gz_threads = atoi(argv[++i]); }
        else if (a == "--cpp-dir") { need(i, 1); c.cpp_dir = argv[++i]; }
        else if (a == "--affine-dir") { need(i, 1); c.affine_dir = argv[++i]; }
        else if (a == "--dump-dir") { need(i, 1); c.dump_dir = argv[++i]; }
        else if (a == "--timing-json") { need(i, 1); c.timing_json = argv[++i]; }
        else if (a == "--verbose") c.verbose = true;
        else fail(2, "unknown option " + a);
    }
    if (c.img.empty() || c.out.empty() || c.plan.empty() || c.atlases.empty()) { usage(); fail(2, "--img, --out, --plan and --atlas are required"); }
    return c;
}

struct Plan {
    int ntissue = 0;
    std::map<int, std::vector<int>> structure;   // label -> tissues
};
Plan read_plan(const std::string &fn)
{
    Plan p;
    FILE *f = fopen(fn.c_str(), "r");
    if (!f) fail(2, "cannot open the plan file " + fn);
    char kw[32];
    while (fscanf(f, "%31s", kw) == 1) {
        if (!strcmp(kw, "tissues")) {
            if (fscanf(f, "%d", &p.ntissue) != 1) fail(2, "bad plan file");
        } else if (!strcmp(kw, "structure")) {
            int label, n;
            if (fscanf(f, "%d %d", &label, &n) != 2) fail(2, "bad plan file");
            std::vector<int> t(n);
            for (int i = 0; i < n; i++) if (fscanf(f, "%d", &t[i]) != 1) fail(2, "bad plan file");
            p.structure[label] = t;
        } else fail(2, "bad plan file: " + std::string(kw));
    }
    fclose(f);
    return p;
}

double since(Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); }

void mkdirs(const std::string &path)
{
    std::string cur;
    std::stringstream ss(path);
    std::string part;
    if (!path.empty() && path[0] == '/') cur = "/";
    while (std::getline(ss, part, '/')) {
        if (part.empty()) continue;
        cur += part + "/";
        mkdir(cur.c_str(), 0777);
    }
}

M44 to_m44(const float x[4][4]) { M44 r; for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) r.m[i][j] = x[i][j]; return r; }

ImgGeom geom_of(const gifio::Image &im) { return make_geom(im.dim[1], im.dim[2], im.dim[3], im.pixdim[1], im.pixdim[2], im.pixdim[3], to_m44(im.xyz)); }

// one output file: header + a host copy of the device data, compressed and written by a background task
struct Writer {
    std::vector<std::future<void>> pending;
    int level, threads;
    void submit(const std::string &fn, std::vector<uint8_t> head, const void *dev, size_t bytes)
    {
        auto buf = std::make_shared<gpucommon::HostBuf>();
        buf->ensure(bytes, 8);
        CK(cudaMemcpy(buf->p, dev, bytes, cudaMemcpyDeviceToHost));
        const int lv = level, th = threads;
        pending.push_back(std::async(std::launch::async, [=]() {
            std::vector<gpucommon::OutFile> files(1);
            files[0].fn = fn;
            files[0].head = head;
            files[0].data = (const uint8_t *)buf->p;
            files[0].nbytes = bytes;
            gpucommon::write_gz_files(files, lv, 4u << 20, th);
        }));
    }
    void wait()
    {
        for (auto &f : pending) f.get();
        pending.clear();
    }
};

}  // namespace

int main(int argc, char **argv)
{
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--check-device")) {   // exit status 0 if a CUDA device can be used
            int n = 0;
            cudaDeviceProp pr;
            if (cudaGetDeviceCount(&n) != cudaSuccess || n < 1 || cudaGetDeviceProperties(&pr, 0) != cudaSuccess) {
                fprintf(stderr, "gif_gpu: no CUDA device available\n");
                return 5;
            }
            printf("%s (sm_%d%d, %.1f GB)\n", pr.name, pr.major, pr.minor, pr.totalGlobalMem / 1e9);
            return 0;
        }
    }
    const Cfg cfg = parse(argc, argv);
    const auto T_START = Clock::now();
    std::map<std::string, double> tm;
    auto lap = [&](const char *name, Clock::time_point t0) { tm[name] = since(t0); if (cfg.verbose) fprintf(stderr, "[%7.2fs] %s %.2f s\n", since(T_START), name, tm[name]); };

    if (cfg.f3d_interp != 1) fail(4, "reg_f3d --interp must be 1 (linear) with the GPU backend");
    if (cfg.f3d_lp > cfg.f3d_ln || cfg.f3d_lp < 1) fail(4, "invalid reg_f3d level settings");
    const int A = (int)cfg.atlases.size();
    if (A > gifgpu::kMaxAtlases) fail(4, "at most " + std::to_string(gifgpu::kMaxAtlases) + " atlases are supported");
    const Plan plan = read_plan(cfg.plan);
    const int T = plan.ntissue;
    if (T < 2 || T > segem::kMaxClasses) fail(4, "the GPU backend supports 2 to " + std::to_string(segem::kMaxClasses) + " tissue classes");
    int dev_count = 0;
    if (cudaGetDeviceCount(&dev_count) != cudaSuccess || dev_count < 1) fail(5, "no CUDA device available");
    CK(cudaFree(0));
    size_t mem_free0 = 0, mem_total = 0;
    CK(cudaMemGetInfo(&mem_free0, &mem_total));
    std::atomic<size_t> mem_peak(0);
    auto note_mem = [&]() {
        size_t fr, to;
        cudaMemGetInfo(&fr, &to);
        size_t used = mem_free0 > fr ? mem_free0 - fr : 0, old = mem_peak.load();
        while (used > old && !mem_peak.compare_exchange_weak(old, used)) {}
    };
    // device memory in use (by this process, relative to the start) is sampled every 10 ms for the report
    std::atomic<bool> sampling(true);
    std::thread sampler([&]() {
        while (sampling.load()) { note_mem(); std::this_thread::sleep_for(std::chrono::milliseconds(10)); }
    });

    // ------------------------------------------------------------------------------------------------ read inputs
    auto t0 = Clock::now();
    gifio::Image target;
    std::vector<gifio::Image> aimg(A), aseg(A);
    {
        std::string err;
        const int nf = 1 + 2 * A;
#pragma omp parallel for schedule(dynamic, 1) num_threads(16)
        for (int k = 0; k < nf; k++) {
            try {
                if (k == 0) gifio::read_image(cfg.img, target);
                else if (k <= A) gifio::read_image(cfg.atlases[k - 1].img, aimg[k - 1]);
                else gifio::read_image(cfg.atlases[k - 1 - A].seg, aseg[k - 1 - A]);
            } catch (std::exception &e) {
#pragma omp critical
                err = e.what();
            }
        }
        if (!err.empty()) fail(3, err);
    }
    lap("read_inputs", t0);
    if (target.dim[0] > 3 && target.dim[4] > 1) fail(4, "the target image must be 3D");
    const ImgGeom rg = geom_of(target);
    const size_t N = rg.nvox();
    ImgGeom fg = geom_of(aimg[0]);
    const size_t Nf = fg.nvox();
    for (int a = 0; a < A; a++) {
        const ImgGeom g = geom_of(aimg[a]), gs = geom_of(aseg[a]);
        for (int k = 0; k < 3; k++) if (g.n[k] != fg.n[k] || gs.n[k] != fg.n[k]) fail(4, "atlas " + cfg.atlases[a].name + ": the atlas images and segmentations must all have the same size (GPU backend)");
        for (int r = 0; r < 4; r++) for (int c = 0; c < 4; c++)
            if (std::fabs(g.xyz.m[r][c] - fg.xyz.m[r][c]) > 1e-4f || std::fabs(gs.xyz.m[r][c] - fg.xyz.m[r][c]) > 1e-4f)
                fail(4, "atlas " + cfg.atlases[a].name + ": all atlases must have the same geometry (GPU backend)");
    }

    // ------------------------------------------------------------------------------------------------ upload, labels
    t0 = Clock::now();
    cudaStream_t st = 0;
    float *d_target = h2d(target.data.data(), N);
    std::vector<float *> d_img(A);
    std::vector<uint16_t *> d_lab(A);
    std::vector<unsigned char *> d_present(A);
    {
        float *d_tmp = dalloc<float>(Nf);
        for (int a = 0; a < A; a++) {
            d_img[a] = h2d(aimg[a].data.data(), Nf);
            CK(cudaMemcpy(d_tmp, aseg[a].data.data(), Nf * sizeof(float), cudaMemcpyHostToDevice));
            d_lab[a] = dalloc<uint16_t>(Nf);
            gifgpu::labels_from_float(d_tmp, d_lab[a], Nf, st);
            d_present[a] = dalloc_zero<unsigned char>(65536);
            gifgpu::label_presence(d_lab[a], Nf, d_present[a], st);
        }
        CK(cudaStreamSynchronize(st));
        pool_free(d_tmp);
    }
    std::set<int> all_labels;
    for (int a = 0; a < A; a++) {
        auto pres = d2h(d_present[a], 65536);
        std::vector<int> bad;
        for (int l = 0; l < 65536; l++) {
            if (!pres[l]) continue;
            all_labels.insert(l);
            if (!plan.structure.count(l)) bad.push_back(l);
        }
        if (!bad.empty()) {
            std::string s;
            for (size_t i = 0; i < bad.size(); i++) s += (i ? ", " : "") + std::to_string(bad[i]);
            fprintf(stderr, "LABELS_NOT_IN_STRUCTURE_INFO\t%s\t%s\t%s\n", cfg.atlases[a].seg.c_str(), cfg.structure_info.c_str(), s.c_str());
            fail(3, "labels of the atlas segmentation " + cfg.atlases[a].seg + " are missing in the structure info");
        }
        pool_free(d_present[a]);
    }
    // class order: ascending labels (gif-python uses the iteration order of a Python set; it only matters for exact ties)
    std::vector<uint16_t> labels(all_labels.begin(), all_labels.end());
    std::vector<std::vector<int>> class_tissues;
    for (uint16_t l : labels) {
        const std::vector<int> &t = plan.structure.at(l);
        for (int x : t) if (x < 0 || x >= T) fail(3, "tissue " + std::to_string(x) + " of label " + std::to_string(l) + " is not in the tissue info");
        class_tissues.push_back(t);
    }
    lap("upload_labels", t0);
    note_mem();

    const int atlas_dt = aimg[0].datatype;   // data type of the floating image (rounding of reg_resample)
    for (int a = 0; a < A; a++) if (aimg[a].datatype != atlas_dt) fail(4, "all atlas images must have the same data type (GPU backend)");

    // ------------------------------------------------------------------------------------------------ registration
    std::vector<M44> affines(A);
    std::vector<float *> d_cpp(A, nullptr);
    GridGeom grid{};
    t0 = Clock::now();
    double t_aladin = 0, t_f3d = 0;
    if (cfg.cpp_dir.empty()) {
        aladin::Params ap;
        ap.levels_to_perform = cfg.aladin_lp;
        aladin::Image tg;
        tg.dev = d_target; tg.nx = rg.n[0]; tg.ny = rg.n[1]; tg.nz = rg.n[2];
        for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) tg.sform.m[i][j] = target.xyz[i][j];
        std::vector<aladin::Image> tl(A);
        for (int a = 0; a < A; a++) {
            tl[a].dev = d_img[a]; tl[a].nx = fg.n[0]; tl[a].ny = fg.n[1]; tl[a].nz = fg.n[2];
            for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) tl[a].sform.m[i][j] = aimg[a].xyz[i][j];
        }
        std::vector<aladin::Mat44> am(A);
        if (cfg.affine_dir.empty()) {
            aladin::AladinBatch ab(tg, ap);
            aladin::TemplateSet ts(tl.data(), A, ap);
            ab.run(ts, am.data());
            for (int a = 0; a < A; a++)
                if (ab.status()[a]) fprintf(stderr, "gif_gpu: warning: reg_aladin failed for atlas %s (too few block correspondences)\n", cfg.atlases[a].name.c_str());
        }
        for (int a = 0; a < A; a++) for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) affines[a].m[i][j] = am[a].m[i][j];
        if (!cfg.affine_dir.empty()) {
            for (int a = 0; a < A; a++) {
                FILE *f = fopen((cfg.affine_dir + "/" + cfg.atlases[a].name + "/affine.txt").c_str(), "r");
                if (!f) fail(3, "cannot read the affine of " + cfg.atlases[a].name);
                for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) if (fscanf(f, "%f", &affines[a].m[i][j]) != 1) fail(3, "bad affine file");
                fclose(f);
            }
        }
        CK(cudaDeviceSynchronize());
        t_aladin = since(t0);
        lap("aladin", t0);
        note_mem();

        t0 = Clock::now();
        F3DParams P;
        P.beWeight = cfg.f3d_be; P.jlWeight = cfg.f3d_jl;
        P.levelNumber = cfg.f3d_ln; P.levelToPerform = cfg.f3d_lp;
        P.maxIterations = cfg.f3d_maxit; P.lnccSigmaMM = cfg.f3d_lncc; P.spacingVox = cfg.f3d_sx;
        P.numerics = cfg.numerics;
        const int chunk = std::max(1, std::min(cfg.f3d_batch, A));
        F3DBatch batch(d_target, rg, nullptr, P, chunk, st);
        grid = batch.finalGrid();
        for (int a = 0; a < A; a++) d_cpp[a] = dalloc<float>(3 * grid.nnodes());
        for (int b0 = 0; b0 < A; b0 += chunk) {
            const int n = std::min(chunk, A - b0);
            std::vector<const float *> tp(n);
            std::vector<float *> out(n);
            for (int k = 0; k < n; k++) { tp[k] = d_img[b0 + k]; out[k] = d_cpp[b0 + k]; }
            std::vector<F3DResult> res(n);
            F3DTiming ft;
            batch.run(tp.data(), fg, n, affines.data() + b0, out.data(), res.data(), &ft);
            note_mem();
        }
        CK(cudaDeviceSynchronize());
        t_f3d = since(t0);
        lap("f3d", t0);
        pool_trim();
        note_mem();
    } else {
        // test mode: control point grids of a CPU run
        t0 = Clock::now();
        for (int a = 0; a < A; a++) {
            gifio::Image g;
            gifio::read_image(cfg.cpp_dir + "/" + cfg.atlases[a].name + "/cpp.nii.gz", g);
            GridGeom gg{};
            for (int k = 0; k < 3; k++) { gg.n[k] = g.dim[1 + k]; gg.d[k] = g.pixdim[1 + k]; }
            gg.xyz = to_m44(g.xyz);
            gg.ijk = m44_inv(gg.xyz);
            if (g.nvox() != 3 * gg.nnodes()) fail(3, "unexpected control point image " + g.path);
            if (a == 0) grid = gg;
            d_cpp[a] = h2d(g.data.data(), g.data.size());
            // the CPU affine is not needed
        }
        lap("load_cpp", t0);
    }

    // ------------------------------------------------------------------------------------------------ resample + weights
    t0 = Clock::now();
    std::vector<float *> d_w(A);
    std::vector<uint16_t *> d_warlab(A);
    {
        gifgpu::Resampler rs(rg, grid, fg, st);
        const float sig[3] = {cfg.lncc_sigma[0], cfg.lncc_sigma[1], cfg.lncc_sigma[2]};
        const float spacing[3] = {target.pixdim[1], target.pixdim[2], target.pixdim[3]};
        gifgpu::LnccWeights lw(d_target, rg.n, sig, spacing, cfg.temperature, st);
        if (!lw.ok()) fail(4, "the LNCC kernel of the fusion weights is too wide for the GPU backend (radius > 40 voxels)");
        float *d_warimg = dalloc<float>(N);
        for (int a = 0; a < A; a++) {
            d_w[a] = dalloc<float>(N);
            d_warlab[a] = dalloc<uint16_t>(N);
            rs.run(d_cpp[a], d_img[a], atlas_dt, d_warimg, d_lab[a], d_warlab[a]);
            lw.run(d_warimg, d_w[a]);
            if (!cfg.dump_dir.empty()) {
                CK(cudaStreamSynchronize(st));
                const std::string dir = cfg.dump_dir + "/" + cfg.atlases[a].name;
                mkdirs(dir);
                int dim[8] = {3, rg.n[0], rg.n[1], rg.n[2], 1, 1, 1, 1};
                float pd[8] = {1, rg.d[0], rg.d[1], rg.d[2], 1, 1, 1, 1};
                gifio::Image dummy;
                Writer w{{}, cfg.gz_level, cfg.gz_threads};
                w.submit(dir + "/weights.nii.gz", gifio::make_header(dim, pd, 16, target.xyz), d_w[a], N * 4);
                w.submit(dir + "/warped_atlas_img_linear_interp.nii.gz", gifio::make_header(dim, pd, 16, target.xyz), d_warimg, N * 4);
                // grid in NiftyReg format
                int gdim[8] = {5, grid.n[0], grid.n[1], grid.n[2], 1, 3, 1, 1};
                float gpd[8] = {1, grid.d[0], grid.d[1], grid.d[2], 1, 1, 1, 1};
                float gx[4][4];
                for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) gx[i][j] = grid.xyz.m[i][j];
                w.submit(dir + "/cpp.nii.gz", gifio::make_header(gdim, gpd, 16, gx, 1007, "NREG_TRANS", 2.f), d_cpp[a], 3 * grid.nnodes() * 4);
                w.wait();
                FILE *f = fopen((dir + "/affine.txt").c_str(), "w");
                for (int i = 0; i < 4; i++) fprintf(f, "%.7f %.7f %.7f %.7f\n", affines[a].m[i][0], affines[a].m[i][1], affines[a].m[i][2], affines[a].m[i][3]);
                fclose(f);
            }
        }
        CK(cudaStreamSynchronize(st));
        pool_free(d_warimg);
    }
    for (int a = 0; a < A; a++) { pool_free(d_img[a]); pool_free(d_lab[a]); pool_free(d_cpp[a]); }
    lap("resample_weights", t0);
    note_mem();

    // ------------------------------------------------------------------------------------------------ outputs: helpers
    mkdirs(cfg.out);
    Writer wr{{}, cfg.gz_level, cfg.gz_threads};
    int dim3[8] = {3, rg.n[0], rg.n[1], rg.n[2], 1, 1, 1, 1};
    int dim4[8] = {4, rg.n[0], rg.n[1], rg.n[2], T, 1, 1, 1};
    float pd3[8] = {1, rg.d[0], rg.d[1], rg.d[2], 1, 1, 1, 1};
    float pd4[8] = {1, rg.d[0], rg.d[1], rg.d[2], 1, 1, 1, 1};

    if (cfg.write_warped_seg) {
        t0 = Clock::now();
        for (int a = 0; a < A; a++) {
            const std::string dir = cfg.out + "/registered_atlases/" + cfg.atlases[a].name;
            mkdirs(dir);
            wr.submit(dir + "/warped_atlas_seg.nii.gz", gifio::make_header(dim3, pd3, 512, target.xyz), d_warlab[a], N * 2);
        }
        lap("warped_seg_to_host", t0);
    }

    // ------------------------------------------------------------------------------------------------ fusion, prior
    t0 = Clock::now();
    int fdims[3] = {rg.n[0], rg.n[1], rg.n[2]};
    gifgpu::Fusion fu(N, fdims, labels, class_tissues, T, st);
    {
        std::vector<const uint16_t *> lp(A);
        std::vector<const float *> wp(A);
        for (int a = 0; a < A; a++) { lp[a] = d_warlab[a]; wp[a] = d_w[a]; }
        fu.set_inputs(lp, wp);
    }
    uint16_t *d_argmax = dalloc<uint16_t>(N);
    float *d_prior = dalloc<float>(N * T);
    fu.argmax_labels(d_argmax);
    fu.tissue_prior(d_prior);
    CK(cudaStreamSynchronize(st));
    wr.submit(cfg.out + "/multi_atlas_proba_seg_argmax.nii.gz", gifio::make_header(dim3, pd3, 512, target.xyz), d_argmax, N * 2);
    wr.submit(cfg.out + "/multi_atlas_tissue_prior.nii.gz", gifio::make_header(dim4, pd4, 16, target.xyz), d_prior, N * T * 4);
    lap("fusion_prior", t0);
    note_mem();

    // ------------------------------------------------------------------------------------------------ seg_EM
    t0 = Clock::now();
    uint8_t *d_tseg = dalloc<uint8_t>(N);
    {
        segem::EmParams ep;
        ep.mrf_beta = cfg.em_mrf_beta; ep.max_iter = cfg.em_maxit; ep.min_iter = cfg.em_minit;
        ep.bias_order = cfg.em_bias_order; ep.bias_ratio = cfg.em_bias_thresh;
        uint8_t *d_mask = dalloc<uint8_t>(N);
        CK(cudaMemset(d_mask, 1, N));
        const unsigned live = segem::EmContext::liveClasses(d_prior, T, N, st);
        segem::EmContext em(rg.n[0], rg.n[1], rg.n[2], T, ep, live, st);
        em.run(d_prior, d_mask, d_target, st, false);
        CK(cudaStreamSynchronize(st));
        fu.tissue_seg(em.seg(), d_prior, d_tseg);
        CK(cudaStreamSynchronize(st));
        wr.submit(cfg.out + "/multi_atlas_tissue_seg.nii.gz", gifio::make_header(dim4, pd4, 16, target.xyz), em.seg(), N * T * 4);
        if (cfg.verbose) fprintf(stderr, "seg_EM: %d iterations\n", em.times().em_iterations);
        pool_free(d_mask);
    }
    lap("seg_em", t0);
    note_mem();

    // ------------------------------------------------------------------------------------------------ final parcellation
    t0 = Clock::now();
    int32_t *d_final = dalloc<int32_t>(N);
    fu.final_parcellation(d_tseg, d_final);
    CK(cudaStreamSynchronize(st));
    wr.submit(cfg.out + "/final_parcellation.nii.gz", gifio::make_header(dim3, pd3, 8, target.xyz), d_final, N * 4);
    lap("final_parcellation", t0);

    t0 = Clock::now();
    wr.wait();
    lap("write_wait", t0);
    tm["total"] = since(T_START);
    sampling = false;
    sampler.join();
    tm["gpu_mem_peak_mb"] = (double)mem_peak.load() / 1e6;
    if (cfg.verbose) fprintf(stderr, "total %.2f s, GPU memory in use (peak, sampled) %.0f MB\n", tm["total"], tm["gpu_mem_peak_mb"]);
    if (!cfg.timing_json.empty()) {
        FILE *f = fopen(cfg.timing_json.c_str(), "w");
        if (f) {
            fprintf(f, "{");
            bool first = true;
            for (auto &kv : tm) { fprintf(f, "%s\"%s\": %.4f", first ? "" : ", ", kv.first.c_str(), kv.second); first = false; }
            fprintf(f, "}\n");
            fclose(f);
        }
    }
    printf("gif_gpu: done in %.2f s\n", tm["total"]);
    return 0;
}
