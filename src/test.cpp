#include "dataset.h"
#include "renderer.h"
#include "allosphere/allosphere_calibration.h"

#include <stdio.h>
#include <string.h>

using namespace allovolume;

// #include <GL/glew.h>
// #include <GLFW/glfw3.h>

// void controls(GLFWwindow* window, int key, int scancode, int action, int mods)
// {
//     if(action == GLFW_PRESS || action == GLFW_REPEAT) {
//         if(key == GLFW_KEY_ESCAPE) {
//             glfwSetWindowShouldClose(window, GL_TRUE);
//         }
//         if(key == GLFW_KEY_W) {
//             phi += 0.1;
//             if(lens) {
//                 lens_origin.x += 1.01e9;
//                 lens->setParameter("origin", &lens_origin);
//                 renderer->render();
//                 img->setNeedsDownload();
//             }
//         }
//         if(key == GLFW_KEY_S) {
//             phi -= 0.1;
//             if(lens) {
//                 lens_origin.x -= 1.01e9;
//                 lens->setParameter("origin", &lens_origin);
//                 renderer->render();
//                 img->setNeedsDownload();
//             }
//         }
//         if(key == GLFW_KEY_A) {
//             theta += 0.1;
//             if(lens) {
//                 lens_origin.y += 1.01e9;
//                 lens->setParameter("origin", &lens_origin);
//                 renderer->render();
//                 img->setNeedsDownload();
//             }
//         }
//         if(key == GLFW_KEY_D) {
//             theta -= 0.1;
//             if(lens) {
//                 lens_origin.y -= 1.01e9;
//                 lens->setParameter("origin", &lens_origin);
//                 renderer->render();
//                 img->setNeedsDownload();
//             }
//         }
//         if(key == GLFW_KEY_Z) {
//             block_index += 1;
//         }
//         if(key == GLFW_KEY_X) {
//             neighbor_index += 1;
//             neighbor_index = neighbor_index % 6;
//             printf("showing neighbor: %d\n", neighbor_index);
//         }
//     }
// }

// GLFWwindow* initWindow(const int resX, const int resY)
// {
//     if(!glfwInit())
//     {
//         fprintf(stderr, "Failed to initialize GLFW\n");
//         return NULL;
//     }
//     glfwWindowHint(GLFW_SAMPLES, 4); // 4x antialiasing

//     // Open a window and create its OpenGL context
//     GLFWwindow* window = glfwCreateWindow(resX, resY, "TEST", NULL, NULL);

//     if(window == NULL)
//     {
//         fprintf(stderr, "Failed to open GLFW window.\n");
//         glfwTerminate();
//         return NULL;
//     }

//     glfwMakeContextCurrent(window);
//     glfwSetKeyCallback(window, controls);

//     // Get info of GPU and supported OpenGL version
//     printf("Renderer: %s\n", glGetString(GL_RENDERER));
//     printf("OpenGL version supported %s\n", glGetString(GL_VERSION));

//     glEnable(GL_DEPTH_TEST); // Depth Testing
//     glDepthFunc(GL_LEQUAL);
//     glDisable(GL_CULL_FACE);
//     glCullFace(GL_BACK);
//     return window;
// }

// float alpha = 0;

// void drawCube(Vector min, Vector max, float r = 1, float g = 1, float b = 1, float a = 1)
// {
//     Vector df = max - min;
//     min += df * 0.05;
//     max -= df * 0.05;
//     GLfloat vertices[] =
//     {
//        -1, -1, -1,   -1, -1,  1,   -1,  1,  1,   -1,  1, -1,
//         1, -1, -1,    1, -1,  1,    1,  1,  1,    1,  1, -1,
//         -1, -1, -1,   -1, -1,  1,    1, -1,  1,    1, -1, -1,
//         -1,  1, -1,   -1,  1,  1,    1,  1,  1,    1,  1, -1,
//         -1, -1, -1,   -1,  1, -1,    1,  1, -1,    1, -1, -1,
//         -1, -1,  1,   -1,  1,  1,    1,  1,  1,    1, -1,  1
//     };
//     for(int i = 0; i < 24; i++) {
//         GLfloat* p = vertices + i * 3;
//         p[0] = p[0] < 0 ? min.x : max.x;
//         p[1] = p[1] < 0 ? min.y : max.y;
//         p[2] = p[2] < 0 ? min.z : max.z;
//     }
//     GLfloat colors[] =
//     {
//         r, g, b, a,  r, g, b, a,  r, g, b, a,  r, g, b, a,
//         r, g, b, a,  r, g, b, a,  r, g, b, a,  r, g, b, a,
//         r, g, b, a,  r, g, b, a,  r, g, b, a,  r, g, b, a,
//         r, g, b, a,  r, g, b, a,  r, g, b, a,  r, g, b, a,
//         r, g, b, a,  r, g, b, a,  r, g, b, a,  r, g, b, a,
//         r, g, b, a,  r, g, b, a,  r, g, b, a,  r, g, b, a
//     };

//     //attempt to rotate cube
//     glPolygonMode( GL_FRONT_AND_BACK, GL_LINE );
//     glEnable(GL_BLEND);
//     glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
//     /* We have a color array and a vertex array */
//     glEnableClientState(GL_VERTEX_ARRAY);
//     glEnableClientState(GL_COLOR_ARRAY);
//     glColorPointer(4, GL_FLOAT, 0, colors);
//     glVertexPointer(3, GL_FLOAT, 0, vertices);

//     /* Send data : 24 vertices */
//     glDrawArrays(GL_QUADS, 0, 24);

//     /* Cleanup states */
//     glDisableClientState(GL_COLOR_ARRAY);
//     glDisableClientState(GL_VERTEX_ARRAY);
// }

// void display( GLFWwindow* window )
// {
//     while(!glfwWindowShouldClose(window))
//     {
//         // Scale to window size
//         GLint windowWidth, windowHeight;
//         glfwGetFramebufferSize(window, &windowWidth, &windowHeight);
//         glViewport(0, 0, windowWidth, windowHeight);

//         // Draw stuff
//         glClearColor(0, 0, 0, 0);
//         glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

//         glMatrixMode(GL_PROJECTION_MATRIX);
//         glLoadIdentity();
//         gluPerspective(60, (double)windowWidth / (double)windowHeight, 0.1, 100 );

//         glMatrixMode(GL_MODELVIEW_MATRIX);
//         gluLookAt(5 * cos(theta) * cos(phi), 5 * sin(theta) * cos(phi), sin(phi) * 5, 0, 0, 0, 0, 0, 1);
//         float scale = 5e19;
//         for(int i = 0; i < volume->getBlockCount(); i++) {
//             if(i != block_index) continue;
//             BlockDescription* desc = volume->getBlockDescription(i);
//             BlockTreeInfo* tinfo = volume->getBlockTreeInfo(i);
//             for(int k = 0; k < 6; k++) {
//                 if(k != neighbor_index) continue;
//                 if(tinfo->neighbors[k] >= 0) {
//                     BlockDescription* desc2 = volume->getBlockDescription(tinfo->neighbors[k]);
//                     drawCube(desc2->min / scale, desc2->max / scale, 0, 1, 0, 0.5);
//                 }
//             }
//             if(tinfo->parent >= 0) {
//                 BlockDescription* desc2 = volume->getBlockDescription(tinfo->parent);
//                 drawCube(desc2->min / scale, desc2->max / scale, 1, 0, 0, 0.5);
//             }
//             drawCube(desc->min / scale, desc->max / scale);
//         }
//         for(int i = 0; i < volume->getBlockCount(); i++) {
//             BlockDescription* desc = volume->getBlockDescription(i);
//             drawCube(desc->min / scale, desc->max / scale, 1, 1, 1, 0.1);
//         }

//         glBegin(GL_LINES);
//         glColor3f(0, 0, 0); glVertex3f(0, 0, 0);
//         glColor3f(1, 0, 0); glVertex3f(1, 0, 0);
//         glColor3f(0, 0, 0); glVertex3f(0, 0, 0);
//         glColor3f(0, 1, 0); glVertex3f(0, 1, 0);
//         glColor3f(0, 0, 0); glVertex3f(0, 0, 0);
//         glColor3f(0, 0, 1); glVertex3f(0, 0, 1);
//         glEnd();


//         // glRasterPos3f(-1, -1, 0);
//         // glDrawPixels(img->getWidth(), img->getHeight(),
//         //     GL_RGBA, GL_FLOAT, img->getPixels());


//         // Update Screen
//         glfwSwapBuffers(window);

//         // Check for any input, or window movement
//         glfwWaitEvents();
//         alpha = 1;
//     }
// }

// #ifndef _WIN32
// #include <sys/time.h>
// double getPreciseTime() {
//     timeval t;
//     gettimeofday(&t, 0);
//     double s = t.tv_sec;
//     s += t.tv_usec / 1000000.0;
//     return s;
// }
// #else
// #include <windows.h>
// double getPreciseTime() {
//     LARGE_INTEGER data, frequency;
//     QueryPerformanceCounter(&data);
//     QueryPerformanceFrequency(&frequency);
//     return (double)data.QuadPart / (double)frequency.QuadPart;
//     //return 0;
// }
// #endif

// float transform_color(float c) {
//     c = (c - 20.0 / 255.0) / (128.0 / 255.0 - 20.0 / 255.0);
//     if(c < 0) c = 0;
//     if(c > 1) c = 1;
//     c = c * c;
//     return c;
// }

// void render_one_frame_as_png_super3d(int argc, char* argv[], bool is_extracted)
// {
//     if(is_extracted) {
//         volume = VolumeBlocks::LoadFromFile(argv[1]);
//     } else {
//         volume = Dataset_FLASH_Create(argv[1], "/dens");
//     }
//     tf = TransferFunction::CreateGaussianTicks(1e-3, 1e8, 20, true);
//     tf->getMetadata()->blend_coefficient = 1e10;
//     // lens_origin = Vector(-0.1e10, 1e8, -1e8);
//     // lens = Lens::CreateEquirectangular(lens_origin, Vector(0, 0, 1), Vector(1, 0, 0));
//     Vector lens1_origin = Vector(0, 2e9, 0);
//     Lens* lens1 = Lens::CreateEquirectangular(lens1_origin, Vector(0, 0, 1), -lens1_origin.normalize());

//     Vector lens2_origin = Vector(0, 0, 2e9);
//     Lens* lens2 = Lens::CreateEquirectangular(lens2_origin, Vector(0, 1, 0), -lens2_origin.normalize());

//     renderer = VolumeRenderer::CreateGPU();

//     int image_size = 1600 * 800;
//     img = Image::Create(1600, 1600);
//     Image* img_render = Image::Create(1600, 800);

//     renderer->setVolume(volume);
//     renderer->setLens(lens1);
//     renderer->setTransferFunction(tf);
//     renderer->setImage(img_render);
//     renderer->render();
//     img_render->setNeedsDownload();
//     Color* pixels = img_render->getPixels();
//     for(int i = 0; i < image_size; i++) {
//         pixels[i].r = transform_color(pixels[i].r);
//         pixels[i].g = transform_color(pixels[i].g);
//         pixels[i].b = transform_color(pixels[i].b);
//         pixels[i].a = 1;
//     }
//     memcpy(img->getPixels(), pixels, sizeof(Color) * image_size);

//     renderer->setLens(lens2);
//     renderer->setTransferFunction(tf);
//     renderer->setImage(img_render);
//     renderer->render();
//     img_render->setNeedsDownload();
//     pixels = img_render->getPixels();
//     for(int i = 0; i < image_size; i++) {
//         pixels[i].r = transform_color(pixels[i].r);
//         pixels[i].g = transform_color(pixels[i].g);
//         pixels[i].b = transform_color(pixels[i].b);
//         pixels[i].a = 1;
//     }
//     memcpy(img->getPixels() + image_size, pixels, sizeof(Color) * image_size);

//     img->save(argv[2], "png16");
// }

// void render_one_frame_as_png_snshock(int argc, char* argv[], bool is_extracted)
// {
//     if(is_extracted) {
//         volume = VolumeBlocks::LoadFromFile(argv[1]);
//     } else {
//         volume = Dataset_FLASH_Create(argv[1], "/dens");
//     }

//     tf = TransferFunction::CreateLinearGradient(1e-25, 1e-22, true);
//     tf->getMetadata()->blend_coefficient = 5e19;
//     // lens_origin = Vector(-0.1e10, 1e8, -1e8);
//     // lens = Lens::CreateEquirectangular(lens_origin, Vector(0, 0, 1), Vector(1, 0, 0));
//     lens_origin = Vector(1.3e18, 1e16, 2e19);
//     lens = Lens::CreateEquirectangular(lens_origin, Vector(0, 1, 0), Vector(0, 0, -1));
//     img = Image::Create(1600, 800);
//     renderer = VolumeRenderer::CreateGPU();
//     renderer->setVolume(volume);
//     renderer->setLens(lens);
//     renderer->setTransferFunction(tf);
//     renderer->setImage(img);

//     renderer->render();
//     img->setNeedsDownload();

//     img->save(argv[2], "png16");
// }

// void render_blocks() {
//     volume = Dataset_FLASH_Create("snshock_3d_hdf5_chk_0266", "/dens");
//     //volume = Dataset_FLASH_Create("super3d_hdf5_plt_cnt_0122", "/dens");
//     GLFWwindow* window = initWindow(800, 400);
//     if( NULL != window )
//     {
//         display( window );
//     }
//     glfwDestroyWindow(window);
//     glfwTerminate();
// }

// void convert_format() {
//     volume = Dataset_FLASH_Create("super3d_hdf5_plt_cnt_0122", "/dens");
//     VolumeBlocks::WriteToFile(volume, "super3d_hdf5_plt_cnt_0122.volume-1");
//     delete volume;
// }

// void speed_test() {
//     volume = VolumeBlocks::LoadFromFile("super3d_hdf5_plt_cnt_0122.volume");
//     tf = TransferFunction::CreateGaussianTicks(1e-3, 1e8, 20, true);
//     tf->getMetadata()->blend_coefficient = 1e10;
//     // lens_origin = Vector(-0.1e10, 1e8, -1e8);
//     // lens = Lens::CreateEquirectangular(lens_origin, Vector(0, 0, 1), Vector(1, 0, 0));
//     Vector lens_origin = Vector(0, 2e9, 0);
//     Lens* lens = Lens::CreateEquirectangular();
//     renderer = VolumeRenderer::CreateGPU();

//     int sizes[] = { 1500, -1 };
//     for(int index = 0; ; index++) {
//         int sz = sizes[index];
//         if(sz < 0) break;
//         Image* img_render = Image::Create(sz, sz / 2);

//         renderer->setVolume(volume);
//         renderer->setLens(lens);
//         renderer->setTransferFunction(tf);
//         renderer->setImage(img_render);

//         for(int i = 0; i < 5; i++) {
//             double t0 = getPreciseTime();
//             renderer->render();
//             img_render->setNeedsDownload();
//             img_render->getPixels()[0].r = 1;

//             double t1 = getPreciseTime();
//             printf("%d x %d, %10.5lf s, %10.5lfus/pixel\n", img_render->getWidth(), img_render->getHeight(), t1 - t0, (t1 - t0) / img_render->getWidth() / img_render->getHeight() * 1e6);
//             if(i == 0) {
//                 char fname[256];
//                 sprintf(fname, "test-%dx%d.png", img_render->getWidth(), img_render->getHeight());
//                 img_render->save(fname, "png16");
//             }
//         }

//         delete img_render;
//     }
// }

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

int main(int argc, char* argv[]) {
    //convert_format();
    //render_one_frame_as_png_super3d(argc, argv, true);
    //render_one_frame_as_png2();
    //render_one_frame_as_png(argc, argv);
    //render_one_frame_as_png2();
    //render_blocks();

    //allosphere_calibration_test();
    super3d_test();
}
