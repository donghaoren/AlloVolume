#include "dataset.h"
#include "renderer.h"
#include <stdio.h>

using namespace allovolume;

#include <GL/glew.h>
#include <GLFW/glfw3.h>

VolumeBlocks* volume;
Image* img;
Lens* lens;
TransferFunction* tf;
VolumeRenderer* renderer;
Vector lens_origin;

int block_index = 0;

void controls(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if(action == GLFW_PRESS || action == GLFW_REPEAT) {
        if(key == GLFW_KEY_ESCAPE)
            glfwSetWindowShouldClose(window, GL_TRUE);
        if(key == GLFW_KEY_W) {
            lens_origin.x += 1.01e9;
            lens->setParameter("origin", &lens_origin);
            renderer->render();
            img->setNeedsDownload();
        }
        if(key == GLFW_KEY_S) {
            lens_origin.x -= 1.01e9;
            lens->setParameter("origin", &lens_origin);
            renderer->render();
            img->setNeedsDownload();
        }
        if(key == GLFW_KEY_A) {
            lens_origin.y += 1.01e9;
            lens->setParameter("origin", &lens_origin);
            renderer->render();
            img->setNeedsDownload();
        }
        if(key == GLFW_KEY_D) {
            lens_origin.y -= 1.01e9;
            lens->setParameter("origin", &lens_origin);
            renderer->render();
            img->setNeedsDownload();
        }
        if(key == GLFW_KEY_Z) {
            block_index += 1;
        }
    }
}

GLFWwindow* initWindow(const int resX, const int resY)
{
    if(!glfwInit())
    {
        fprintf(stderr, "Failed to initialize GLFW\n");
        return NULL;
    }
    glfwWindowHint(GLFW_SAMPLES, 4); // 4x antialiasing

    // Open a window and create its OpenGL context
    GLFWwindow* window = glfwCreateWindow(resX, resY, "TEST", NULL, NULL);

    if(window == NULL)
    {
        fprintf(stderr, "Failed to open GLFW window.\n");
        glfwTerminate();
        return NULL;
    }

    glfwMakeContextCurrent(window);
    glfwSetKeyCallback(window, controls);

    // Get info of GPU and supported OpenGL version
    printf("Renderer: %s\n", glGetString(GL_RENDERER));
    printf("OpenGL version supported %s\n", glGetString(GL_VERSION));

    glEnable(GL_DEPTH_TEST); // Depth Testing
    glDepthFunc(GL_LEQUAL);
    glDisable(GL_CULL_FACE);
    glCullFace(GL_BACK);
    return window;
}

float alpha = 0;

void drawCube(Vector min, Vector max)
{
    Vector df = Vector(1,1,1);
    min += df * 0.05;
    max -= df * 0.05;
    GLfloat vertices[] =
    {
       -1, -1, -1,   -1, -1,  1,   -1,  1,  1,   -1,  1, -1,
        1, -1, -1,    1, -1,  1,    1,  1,  1,    1,  1, -1,
        -1, -1, -1,   -1, -1,  1,    1, -1,  1,    1, -1, -1,
        -1,  1, -1,   -1,  1,  1,    1,  1,  1,    1,  1, -1,
        -1, -1, -1,   -1,  1, -1,    1,  1, -1,    1, -1, -1,
        -1, -1,  1,   -1,  1,  1,    1,  1,  1,    1, -1,  1
    };
    for(int i = 0; i < 24; i++) {
        GLfloat* p = vertices + i * 3;
        p[0] = p[0] < 0 ? min.x : max.x;
        p[1] = p[1] < 0 ? min.y : max.y;
        p[2] = p[2] < 0 ? min.z : max.z;
    }
    float a = 0.5;
    GLfloat colors[] =
    {
        1, 1, 1, a,  1, 1, 1, a,  1, 1, 1, a,  1, 1, 1, a,
        1, 1, 1, a,  1, 1, 1, a,  1, 1, 1, a,  1, 1, 1, a,
        1, 1, 1, a,  1, 1, 1, a,  1, 1, 1, a,  1, 1, 1, a,
        1, 1, 1, a,  1, 1, 1, a,  1, 1, 1, a,  1, 1, 1, a,
        1, 1, 1, a,  1, 1, 1, a,  1, 1, 1, a,  1, 1, 1, a,
        1, 1, 1, a,  1, 1, 1, a,  1, 1, 1, a,  1, 1, 1, a
    };

    //attempt to rotate cube
    glPolygonMode( GL_FRONT_AND_BACK, GL_LINE );
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    /* We have a color array and a vertex array */
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);
    glColorPointer(4, GL_FLOAT, 0, colors);
    glVertexPointer(3, GL_FLOAT, 0, vertices);

    /* Send data : 24 vertices */
    glDrawArrays(GL_QUADS, 0, 24);

    /* Cleanup states */
    glDisableClientState(GL_COLOR_ARRAY);
    glDisableClientState(GL_VERTEX_ARRAY);
}

void display( GLFWwindow* window )
{
    while(!glfwWindowShouldClose(window))
    {
        // Scale to window size
        GLint windowWidth, windowHeight;
        glfwGetWindowSize(window, &windowWidth, &windowHeight);
        glViewport(0, 0, windowWidth, windowHeight);

        // Draw stuff
        glClearColor(0, 0, 0, 0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // glMatrixMode(GL_PROJECTION_MATRIX);
        // glLoadIdentity();
        // gluPerspective(60, (double)windowWidth / (double)windowHeight, 0.1, 100 );

        // glMatrixMode(GL_MODELVIEW_MATRIX);
        // gluLookAt(5, 4, 3, 0, 0, 0, 0, 0, 1);
        // glRotatef(alpha, 0, 1, 0);

        // for(int i = 0; i < volume->getBlockCount(); i++) {
        //     if(i != block_index) continue;
        //     BlockDescription* desc = volume->getBlockDescription(i);
        //     drawCube(desc->min / 1e10, desc->max / 1e10);
        // }


        glRasterPos3f(-1, -1, 0);
        glDrawPixels(img->getWidth(), img->getHeight(),
            GL_RGBA, GL_FLOAT, img->getPixels());


        // Update Screen
        glfwSwapBuffers(window);

        // Check for any input, or window movement
        glfwWaitEvents();
        alpha = 1;
    }
}

int main(int argc, char** argv)
{
    volume = Dataset_FLASH_Create("super3d_hdf5_plt_cnt_0122", "/dens");
    tf = TransferFunction::CreateTest();
    lens_origin = Vector(-0.4e10, 1e8, -1e8);
    lens = Lens::CreateEquirectangular(lens_origin, Vector(0, 0, 1), Vector(1, 0, 0));
    // lens_origin = Vector(0, 0, 1e10);
    // lens = Lens::CreateEquirectangular(lens_origin, Vector(0, 1, 0), Vector(0, 0, -1));
    img = Image::Create(1600, 800);
    renderer = VolumeRenderer::CreateGPU();
    renderer->setVolume(volume);
    renderer->setLens(lens);
    renderer->setTransferFunction(tf);
    renderer->setImage(img);

    renderer->render();
    img->setNeedsDownload();

    GLFWwindow* window = initWindow(800, 400);
    if( NULL != window )
    {
        display( window );
    }
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}

// int main() {
//     volume = Dataset_FLASH_Create("/Users/donghao/super3d_hdf5_plt_cnt_0122", "/dens");
//     for(int i = 0; i < volume->getBlockCount(); i++) {
//         BlockDescription* desc = volume->getBlockDescription(i);
//         Vector ff = desc->max - desc->min;
//         printf("%g %g %g - %g %g %g - %g %g %g\n", desc->min.x, desc->min.y, desc->min.z,
//             desc->max.x, desc->max.y, desc->max.z, ff.x, ff.y, ff.z);
//         double sum = 0;
//         for(int k = 0; k < 32 * 32 * 32; k++) sum += volume->getData()[desc->offset + k];
//         sum /= 32 * 32 * 32;
//     printf("%lf\n", sum);
//     }

//     // TransferFunction* tf = TransferFunction::CreateTest();
//     // Lens* lens = Lens::CreateEquirectangular(Vector(0, 0, 0), Vector(0, 0, 1), Vector(1, 0, 0));
//     // Image* img = Image::Create(100, 100);

//     // VolumeRenderer* renderer = VolumeRenderer::CreateGPU();

//     // renderer->render();

//     delete volume;
// }
