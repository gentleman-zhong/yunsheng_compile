#include "popsift.h"
#include <algorithm>
#include <cmath>
#include <mutex>

namespace pps{

PopSiftContext *ctx = nullptr;
std::mutex g_mutex;

PopSiftContext::PopSiftContext() : ps(nullptr){
    // Check if CUDA is working
    int currentDevice;
    if (cudaGetDevice( &currentDevice ) != 0){
        // Try resetting
        cudaDeviceReset();

        cudaError_t err;
        if ((err = cudaGetDevice( &currentDevice )) != 0){
            throw std::runtime_error("Cannot use CUDA device: " + std::string(cudaGetErrorString(err)));
        }
    }

    popsift::cuda::device_prop_t deviceInfo;
    deviceInfo.set(0, false);
    config = new popsift::Config();

    // filter_max_extrema 已在类声明时默认为 20000
}

PopSiftContext::~PopSiftContext(){
    delete ps;
    ps = nullptr;
    delete config;
    config = nullptr;
}

void PopSiftContext::setup(float peak_threshold, float edge_threshold, bool use_root, float downsampling, int filter_max_extrema){
    bool changed = false;
    if (this->peak_threshold != peak_threshold) { this->peak_threshold = peak_threshold; changed = true; }
    if (this->edge_threshold != edge_threshold) { this->edge_threshold = edge_threshold; changed = true; }
    if (this->use_root != use_root) { this->use_root = use_root; changed = true; }
    if (this->downsampling != downsampling) { this->downsampling = downsampling; changed = true; }

    // 规范 filter 参数：如果传入 <= 0，使用默认 20000
    if (filter_max_extrema <= 0) filter_max_extrema = 20000;

    if (this->filter_max_extrema != filter_max_extrema) { this->filter_max_extrema = filter_max_extrema; changed = true; }

    if (changed){
        config->setThreshold(peak_threshold);
        config->setEdgeLimit(edge_threshold);
        config->setNormMode(use_root ? popsift::Config::RootSift : popsift::Config::Classic );
        config->setFilterSorting(popsift::Config::LargestScaleFirst);
        config->setMode(popsift::Config::OpenCV);
        config->setDownsampling(downsampling);
        // config->setOctaves(4);
        // config->setLevels(3);

        // 确保向底层设置为合理正值（至少 1）
        int cfg_val = std::max(1, this->filter_max_extrema);
        config->setFilterMaxExtrema(cfg_val);

        if (ps){
            delete ps;
            ps = nullptr;
        }
        ps = new PopSift(*config,
                    popsift::Config::ProcessingMode::ExtractingMode,
                    PopSift::ByteImages );
    }
}

PopSift *PopSiftContext::get(){
    return ps;
}

py::object popsift(pyarray_uint8 image,
                 float peak_threshold,
                 float edge_threshold,
                 int target_num_features,
                 bool use_root,
                 float downsampling,
                 int filter_max_extrema) {
    if (!image.size()) return py::none();

    py::gil_scoped_release release;

    int width = image.shape(1);
    int height = image.shape(0);
    int numFeatures = 0;

    // 防御：如果传入 <= 0，则采用 20000
    if (filter_max_extrema <= 0) filter_max_extrema = 20000;

    // effective_target = min(请求的 target, filter 限制)
    int effective_target = std::min(target_num_features, filter_max_extrema);

    while(true){
        g_mutex.lock();
        if (!ctx) ctx = new PopSiftContext();
        // 传入 filter_max_extrema 到 setup
        ctx->setup(peak_threshold, edge_threshold, use_root, downsampling, filter_max_extrema);
        std::unique_ptr<SiftJob> job(ctx->get()->enqueue( width, height, image.data() ));
        std::unique_ptr<popsift::Features> result(job->get());
        g_mutex.unlock();

        numFeatures = result->getFeatureCount();

        // 使用 effective_target 判断是否达到期望（而不是直接使用 target_num_features）
        if (numFeatures >= effective_target || peak_threshold < 0.0001){
            popsift::Feature* feature_list = result->getFeatures();
            std::vector<float> points(4 * numFeatures);
            std::vector<float> desc(128 * numFeatures);

            for (size_t i = 0; i < numFeatures; i++){
                popsift::Feature pFeat = feature_list[i];

                for(int oriIdx = 0; oriIdx < pFeat.num_ori; oriIdx++){
                    const popsift::Descriptor* pDesc = pFeat.desc[oriIdx];

                    for (int k = 0; k < 128; k++){
                        desc[128 * i + k] = pDesc->features[k];
                    }

                    points[4 * i + 0] = std::min<float>(std::round(pFeat.xpos), width - 1);
                    points[4 * i + 1] = std::min<float>(std::round(pFeat.ypos), height - 1);
                    points[4 * i + 2] = pFeat.sigma;
                    points[4 * i + 3] = pFeat.orientation[oriIdx];
                }
            }

            py::gil_scoped_acquire acquire;
            py::list retn;
            retn.append(py_array_from_data(&points[0], numFeatures, 4));
            retn.append(py_array_from_data(&desc[0], numFeatures, 128));
            return retn;
        }else{
           // Lower peak threshold if we don't meet the effective_target
           peak_threshold = (peak_threshold * 2.0) / 3.0;
        }
    }

    // We should never get here
    py::gil_scoped_acquire acquire;
    return py::none();
}

bool fitsTexture(int width, int height, float downsampling){
    if (!ctx) ctx = new PopSiftContext();
    // 同步：传入默认 filter_max_extrema（20000）
    ctx->setup(0.06, 10, true, downsampling, 20000);

    PopSift::AllocTest a = ctx->get()->testTextureFit( width, height );
    return a == PopSift::AllocTest::Ok;
}

} 
