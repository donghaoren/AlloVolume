#include "dataset.h"
#include "renderer.h"
#include "allosphere/allosphere_calibration.h"

#include <stdio.h>
#include <string.h>

using namespace allovolume;

void convert_to_grayscale(Color& c) {
    float gray = 0.2989 * c.r + 0.5870 * c.g + 0.1140 * c.b;
    c.r = c.g = c.b = gray;
}

#include <sys/time.h>
double getPreciseTime() {
    timeval t;
    gettimeofday(&t, 0);
    double s = t.tv_sec;
    s += t.tv_usec / 1000000.0;
    return s;
}

Pose pose_lookat(Vector eye, Vector lookat, Vector up) {
    Pose pose;
    pose.position = eye;
    Vector direction = lookat - eye;
    pose.rotation = Quaternion::Rotation(direction.normalize().cross(Vector(1, 0, 0)), -acos(direction.normalize().dot(Vector(1, 0, 0))));
    return pose;
}

void super3d_test() {
    VolumeBlocks* volume = VolumeBlocks::LoadFromFile("super3d_hdf5_plt_cnt_0122.volume");
    TransferFunction* tf = TransferFunction::CreateGaussianTicks(1e-3, 1e8, TransferFunction::kLogScale, 32);
    Pose pose;
    pose.position = Vector(-1e10, 0, -0.1e10) * 3;
    pose.rotation = Quaternion::Rotation(-pose.position.normalize().cross(Vector(1, 0, 0)), -acos(-pose.position.normalize().dot(Vector(1, 0, 0))));

    //Lens* lens = Lens::CreateEquirectangular();
    Lens* lens = Lens::CreatePerspective(PI / 2.0);

    VolumeRenderer* renderer = VolumeRenderer::CreateGPU();
    renderer->setPose(pose);

    renderer->setVolume(volume);
    renderer->setLens(lens);
    renderer->setTransferFunction(tf);
    renderer->setBlendingCoefficient(1e10);
    float sz = 2.5e10;
    renderer->setBoundingBox(Vector(-sz, -sz, -sz), Vector(+sz, +sz, +sz));
    //renderer->setRaycastingMethod(VolumeRenderer::kAdaptiveRKVMethod);
    renderer->setRaycastingMethod(VolumeRenderer::kRK4Method);

    int width = 400, height = 400;

    float radius = pose.position.len();
    float eye_separation = radius / 10.0;

    lens->setFocalDistance(radius);
    lens->setEyeSeparation(+eye_separation);
    Image* img_render_left = Image::Create(width, height);
    renderer->setImage(img_render_left);
    renderer->render();
    img_render_left->setNeedsDownload();
    img_render_left->save("super3d_left.png", "png16");

    lens->setFocalDistance(radius);
    lens->setEyeSeparation(-eye_separation);
    Image* img_render_right = Image::Create(width, height);
    renderer->setImage(img_render_right);
    double t0 = getPreciseTime();
    renderer->render();
    double t1 = getPreciseTime();
    printf("Time: %.2lf\n", (t1 - t0) * 1000.0);
    img_render_right->setNeedsDownload();
    img_render_right->save("super3d_right.png", "png16");

    Image* img_render = Image::Create(width, height);
    Color* dest = img_render->getPixels();
    Color* sleft = img_render_left->getPixels();
    Color* sright = img_render_right->getPixels();
    for(int i = 0; i < width * height; i++) {
        convert_to_grayscale(sleft[i]);
        convert_to_grayscale(sright[i]);
        dest[i] = Color(sleft[i].r, sright[i].g, sright[i].b, 1);
    }

    img_render->save("super3d_ana.png", "png16");

    delete img_render_left;
    delete img_render_right;
    delete img_render;
    delete renderer;
    delete lens;
    delete tf;
    delete volume;
}

void super3d_closeup_cell_boundary() {
    VolumeBlocks* volume = VolumeBlocks::LoadFromFile("super3d_hdf5_plt_cnt_0122.volume");
    TransferFunction* tf = TransferFunction::CreateGaussianTicks(1e4, 1e8, TransferFunction::kLogScale, 16);
    //Lens* lens = Lens::CreateEquirectangular();
    Lens* lens = Lens::CreatePerspective(PI / 2.0);

    VolumeRenderer* renderer = VolumeRenderer::CreateGPU();

    renderer->setVolume(volume);
    renderer->setLens(lens);
    renderer->setTransferFunction(tf);
    renderer->setBlendingCoefficient(1e8);
    float sz = 2.5e10;
    renderer->setBoundingBox(Vector(-sz, -sz, -sz), Vector(+sz, +sz, +sz));
    renderer->setRaycastingMethod(VolumeRenderer::kAdaptiveRKVMethod);
    //renderer->setRaycastingMethod(VolumeRenderer::kRK4Method);

    int width = 600, height = 400;

    Image* img_render = Image::Create(width, height);

    renderer->setImage(img_render);

    { // Front
        Pose pose;
        pose.position = Vector(-1e9, 0, 0);
        pose.rotation = Quaternion(1, Vector(0, 0, 0));
        renderer->setPose(pose);
        renderer->render();
        img_render->setNeedsDownload();
        img_render->save("super3d_closeup_cell_boundary_front.png", "png16");
    }
    { // Bottom-Top
        Pose pose;
        pose.position = Vector(0, 0, -1e9);
        pose.rotation = Quaternion::Rotation(Vector(0, 1, 0), -PI / 2);
        renderer->setPose(pose);
        renderer->render();
        img_render->setNeedsDownload();
        img_render->save("super3d_closeup_cell_boundary_bottom.png", "png16");
    }

    delete img_render;
    delete renderer;
    delete lens;
    delete tf;
    delete volume;
}

int min(int a, int b) { return a < b ? a : b; }

struct rgb_curve_t {
    float vmin;
    float vmax;
    rgb_curve_t(float min_, float max_) : vmin(min_), vmax(max_) { }
    Color operator () (const Color& c) {
        Color r;
        r.r = clamp01((c.r - vmin) / (vmax - vmin));
        r.g = clamp01((c.g - vmin) / (vmax - vmin));
        r.b = clamp01((c.b - vmin) / (vmax - vmin));
        r.a = c.a;
        return r;
    }
};

template <typename CurveT>
void blocked_rendering(VolumeRenderer* renderer, int width, int height, const char* output, CurveT curve) {
    int bw = 400;
    int bh = 400;
    Color* total_data = new Color[width * height];
    for(int x = 0; x < width; x += bw) {
        for(int y = 0; y < height; y += bh) {
            int w = min(bw, width - x);
            int h = min(bh, height - y);
            Image* block_data = Image::Create(w, h);
            renderer->setImage(block_data);
            renderer->render(x, y, width, height);
            block_data->setNeedsDownload();
            Color* pixels = block_data->getPixels();
            for(int ty = 0; ty < h; ty++) {
                for(int tx = 0; tx < w; tx++) {
                    total_data[(ty + y) * width + (tx + x)] = curve(pixels[ty * w + tx]);
                }
            }
            delete block_data;
        }
    }
    Image::WriteImageFile(output, "png16", width, height, total_data);
    printf("Written to: %s\n", output);
    delete [] total_data;
}

void super3d_render_volume(int index_min, int index_max) {
    TransferFunction* tf_close = TransferFunction::CreateGaussianTicks(1e4, 1e8, TransferFunction::kLogScale, 16);
    TransferFunction* tf_far = TransferFunction::CreateGaussianTicks(1e-3, 1e8, TransferFunction::kLogScale, 16);
    //Lens* lens = Lens::CreateEquirectangular();
    Lens* lens = Lens::CreatePerspective(PI / 2.0);

    VolumeRenderer* renderer = VolumeRenderer::CreateGPU();

    renderer->setLens(lens);

    float sz = 2.5e10;
    renderer->setBoundingBox(Vector(-sz, -sz, -sz), Vector(+sz, +sz, +sz));

    renderer->setRaycastingMethod(VolumeRenderer::kAdaptiveRKVMethod);
    //renderer->setRaycastingMethod(VolumeRenderer::kRK4Method);

    int width = 1920, height = 1280;
    renderer->setBackgroundColor(Color(0, 0, 0, 1));

    for(int index = index_min; index <= index_max; index++) {
        char filename[256], output_filename[256];
        sprintf(filename, "/data/donghao/dataset_flash/super3d_hdf5_plt/density/super3d_hdf5_plt_cnt_%04d.volume", index);
        VolumeBlocks* volume = VolumeBlocks::LoadFromFile(filename);

        renderer->setVolume(volume);
        { // Front
            Pose pose;
            pose.position = Vector(-1e9, 0, 0);
            pose.rotation = Quaternion(1, Vector(0, 0, 0));
            renderer->setPose(pose);
            renderer->setTransferFunction(tf_close);
            renderer->setBlendingCoefficient(1e8);

            sprintf(output_filename, "super3d/close-front/frame%04d.png", index);
            blocked_rendering(renderer, width, height, output_filename, rgb_curve_t(0, 1));
        }
        { // Bottom-Top
            Pose pose;
            pose.position = Vector(0, 0, -1e9);
            pose.rotation = Quaternion::Rotation(Vector(0, 1, 0), -PI / 2);
            renderer->setPose(pose);
            renderer->setTransferFunction(tf_close);
            renderer->setBlendingCoefficient(1e8);

            sprintf(output_filename, "super3d/close-bottom/frame%04d.png", index);
            blocked_rendering(renderer, width, height, output_filename, rgb_curve_t(0, 1));
        }
        { // Front far
            Pose pose;
            pose.position = Vector(-1e10, 0, 0);
            pose.rotation = Quaternion(1, Vector(0, 0, 0));
            renderer->setPose(pose);
            renderer->setTransferFunction(tf_far);
            renderer->setBlendingCoefficient(1e9);

            sprintf(output_filename, "super3d/far-front/frame%04d.png", index);
            blocked_rendering(renderer, width, height, output_filename, rgb_curve_t(0, 0.7));
        }
        { // Bottom far
            Pose pose;
            pose.position = Vector(0, 0, -1e10);
            pose.rotation = Quaternion::Rotation(Vector(0, 1, 0), -PI / 2);
            renderer->setPose(pose);
            renderer->setTransferFunction(tf_far);

            sprintf(output_filename, "super3d/far-bottom/frame%04d.png", index);
            blocked_rendering(renderer, width, height, output_filename, rgb_curve_t(0, 0.5));
        }
        delete volume;
    }

    delete renderer;
    delete lens;
    delete tf_far;
    delete tf_close;
}

void snshock_performance_test() {
    VolumeBlocks* volume = VolumeBlocks::LoadFromFile("snshock_3d_hdf5_chk_0248.volume");
    VolumeRenderer* renderer = VolumeRenderer::CreateGPU();
    Lens* lens = Lens::CreatePerspective(PI / 2.0);
    Pose pose;
    pose.position = Vector(0, 0, -10e19);
    pose.rotation = Quaternion::Rotation(Vector(0, 1, 0), -PI / 2);
    renderer->setPose(pose);
    TransferFunction* tf_far = TransferFunction::CreateLinearGradient(1e-25, 1e-22, TransferFunction::kLogScale);
    renderer->setTransferFunction(tf_far);

    Image* img = Image::Create(760, 700);

    renderer->setLens(lens);
    renderer->setVolume(volume);
    renderer->setImage(img);
    renderer->setBlendingCoefficient(5e19);
    renderer->setBackgroundColor(Color(0, 0, 0, 1));

    printf("RK4:\n");
    renderer->setRaycastingMethod(VolumeRenderer::kRK4Method);
    for(int i = 0; i < 5; i++) {
        double t0 = getPreciseTime();
        renderer->render();
        double t1 = getPreciseTime();
        printf("  Size: %dx%d, Time: %.3lf ms, FPS = %.3lf\n", img->getWidth(), img->getHeight(), (t1 - t0) * 1000, 1.0 / (t1 - t0));
    }
    img->setNeedsDownload();
    img->save("snshock_performance_test_rk4.png", "png16");

    printf("Basic:\n");
    renderer->setRaycastingMethod(VolumeRenderer::kBasicBlendingMethod);
    for(int i = 0; i < 5; i++) {
        double t0 = getPreciseTime();
        renderer->render();
        double t1 = getPreciseTime();
        printf("  Size: %dx%d, Time: %.3lf ms, FPS = %.3lf\n", img->getWidth(), img->getHeight(), (t1 - t0) * 1000, 1.0 / (t1 - t0));
    }
    img->setNeedsDownload();
    img->save("snshock_performance_test_basic.png", "png16");
}

void super3d_highres() {
    VolumeBlocks* volume = VolumeBlocks::LoadFromFile("super3d_hdf5_plt_cnt_0122.volume");
    VolumeRenderer* renderer = VolumeRenderer::CreateGPU();
    Lens* lens = Lens::CreatePerspective(PI / 2.0);
    Pose pose = pose_lookat(Vector(2e10, 0, 1e10) / 5, Vector(0, 0, 0), Vector(0, 0, 1));
    renderer->setPose(pose);
    TransferFunction* tf_far = TransferFunction::CreateGaussianTicks(1e2, 1e8, TransferFunction::kLogScale, 16);
    renderer->setTransferFunction(tf_far);
    renderer->setLens(lens);
    renderer->setVolume(volume);
    renderer->setBlendingCoefficient(3e8);
    renderer->setBackgroundColor(Color(0, 0, 0, 1));

    blocked_rendering(renderer, 4000, 4000, "super3d_highres_close.png", rgb_curve_t(0, 1));
}

void super3d_highres_stereo() {
    VolumeBlocks* volume = VolumeBlocks::LoadFromFile("super3d_hdf5_plt_cnt_0122.volume");
    VolumeRenderer* renderer = VolumeRenderer::CreateGPU();
    Lens* lens = Lens::CreatePerspective(PI / 2.0);
    Pose pose = pose_lookat(Vector(2e10, 0, 1e10), Vector(0, 0, 0), Vector(0, 0, 1));
    renderer->setPose(pose);
    TransferFunction* tf_far = TransferFunction::CreateGaussianTicks(1e-2, 1e5, TransferFunction::kLogScale, 16);
    renderer->setTransferFunction(tf_far);
    renderer->setLens(lens);
    renderer->setVolume(volume);
    renderer->setBlendingCoefficient(1e9);
    renderer->setBackgroundColor(Color(0, 0, 0, 1));

    lens->setFocalDistance(pose.position.len());

    lens->setEyeSeparation(pose.position.len() / 10);
    blocked_rendering(renderer, 1000, 1000, "super3d_highres_close_left.png", rgb_curve_t(0, 1));

    lens->setEyeSeparation(-pose.position.len() / 10);
    blocked_rendering(renderer, 1000, 1000, "super3d_highres_close_right.png", rgb_curve_t(0, 1));
}

void super3d_performance_test() {
    VolumeBlocks* volume = VolumeBlocks::LoadFromFile("super3d_hdf5_plt_cnt_0122.volume");
    VolumeRenderer* renderer = VolumeRenderer::CreateGPU();
    Lens* lens = Lens::CreatePerspective(PI / 2.0);
    Pose pose = pose_lookat(Vector(2e10, 0, 0.3e10), Vector(0, 0, 0), Vector(0, 0, 1));
    renderer->setPose(pose);
    TransferFunction* tf_far = TransferFunction::CreateGaussianTicks(1e-2, 1e5, TransferFunction::kLogScale, 12);
    renderer->setTransferFunction(tf_far);

    Image* img = Image::Create(1024, 1024);

    renderer->setLens(lens);
    renderer->setVolume(volume);
    renderer->setImage(img);
    renderer->setBlendingCoefficient(1e9);
    renderer->setBackgroundColor(Color(0, 0, 0, 1));
    float sz = 2.5e10;
    renderer->setBoundingBox(Vector(-sz, -sz, -sz), Vector(+sz, +sz, +sz));

    printf("RK4:\n");
    renderer->setRaycastingMethod(VolumeRenderer::kRK4Method);
    for(int i = 0; i < 5; i++) {
        double t0 = getPreciseTime();
        renderer->render();
        double t1 = getPreciseTime();
        printf("  Size: %dx%d, Time: %.3lf ms, FPS = %.3lf\n", img->getWidth(), img->getHeight(), (t1 - t0) * 1000, 1.0 / (t1 - t0));
    }
    img->setNeedsDownload();
    img->save("super3d_performance_test_rk4.png", "png16");

    printf("Basic:\n");
    renderer->setRaycastingMethod(VolumeRenderer::kBasicBlendingMethod);
    for(int i = 0; i < 5; i++) {
        double t0 = getPreciseTime();
        renderer->render();
        double t1 = getPreciseTime();
        printf("  Size: %dx%d, Time: %.3lf ms, FPS = %.3lf\n", img->getWidth(), img->getHeight(), (t1 - t0) * 1000, 1.0 / (t1 - t0));
    }
    img->setNeedsDownload();
    img->save("super3d_performance_test_basic.png", "png16");

    // printf("Adaptive RKV:\n");
    // renderer->setRaycastingMethod(VolumeRenderer::kAdaptiveRKVMethod);
    // for(int i = 0; i < 5; i++) {
    //     double t0 = getPreciseTime();
    //     renderer->render();
    //     double t1 = getPreciseTime();
    //     printf("  Size: %dx%d, Time: %.3lf ms, FPS = %.3lf\n", img->getWidth(), img->getHeight(), (t1 - t0) * 1000, 1.0 / (t1 - t0));
    // }
    // img->setNeedsDownload();
    // img->save("super3d_performance_test_rkv.png", "png16");

}

void kdtree_test() {
    VolumeBlocks* volume = VolumeBlocks::LoadFromFile("super3d_hdf5_plt_cnt_0122.volume");
    VolumeRenderer* renderer = VolumeRenderer::CreateGPU();
    Lens* lens = Lens::CreatePerspective(PI / 2.0);
    Pose pose = pose_lookat(Vector(2e10, 0, 0.3e10), Vector(0, 0, 0), Vector(0, 0, 1));
    renderer->setPose(pose);
    TransferFunction* tf_far = TransferFunction::CreateGaussianTicks(1e-2, 1e5, TransferFunction::kLogScale, 12);
    renderer->setTransferFunction(tf_far);

    Image* img = Image::Create(1024, 1024);

    renderer->setLens(lens);
    renderer->setVolume(volume);
    renderer->setImage(img);
    renderer->setBlendingCoefficient(1e9);
    renderer->setBackgroundColor(Color(0, 0, 0, 1));
}

int main(int argc, char* argv[]) {
    //convert_format();
    //render_one_frame_as_png_super3d(argc, argv, true);
    //render_one_frame_as_png2();
    //render_one_frame_as_png(argc, argv);
    //render_one_frame_as_png2();
    //render_blocks();

    //allosphere_calibration_test();
    super3d_performance_test();
    //super3d_highres();
    //super3d_highres_stereo();
    //kdtree_test();
    //snshock_performance_test();
    //super3d_render_volume(atoi(argv[1]), atoi(argv[1]));
}
