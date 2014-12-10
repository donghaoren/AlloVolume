// Test glfw.

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <stdio.h>
#include <math.h>

int max(int a, int b) { return a > b ? a : b; }

int main() {
    glfwInit();
    int monitor_count = 0;
    GLFWmonitor** monitors = glfwGetMonitors(&monitor_count);

    // Get the size of the virtual screen.
    int screen_width = 0, screen_height = 0;
    for(int i = 0; i < monitor_count; i++) {
        int x, y;
        const GLFWvidmode *mode = glfwGetVideoMode(monitors[i]);
        glfwGetMonitorPos(monitors[i], &x, &y);
        printf("Monitor: %d %d, %d x %d\n", x, y, mode->width, mode->height);
        x += mode->width;
        y += mode->height;
        screen_width = max(screen_width, x);
        screen_height = max(screen_height, y);
    }
    printf("Screen: %d %d\n", screen_width, screen_height);

    glfwWindowHint(GLFW_STEREO, GL_TRUE);
    GLFWwindow* window = glfwCreateWindow(800, 600, "Allosphere Volume Renderer", NULL, NULL);
    if(window) {
        printf("Successful!\n");
        glfwMakeContextCurrent(window);

        while(!glfwWindowShouldClose(window)) {
            float ratio;
            int width, height;
            glfwGetFramebufferSize(window, &width, &height);
            ratio = width / (float) height;
            glViewport(0, 0, width, height);
            glClear(GL_COLOR_BUFFER_BIT);
            glMatrixMode(GL_PROJECTION);
            glLoadIdentity();
            glOrtho(-ratio, ratio, -1.f, 1.f, 1.f, -1.f);
            glMatrixMode(GL_MODELVIEW);
            glLoadIdentity();
            glRotatef((float) glfwGetTime() * 50.f, 0.f, 0.f, 1.f);
            glBegin(GL_TRIANGLES);
            glColor3f(1.f, 0.f, 0.f);
            glVertex3f(-0.6f, -0.4f, 0.f);
            glColor3f(0.f, 1.f, 0.f);
            glVertex3f(0.6f, -0.4f, 0.f);
            glColor3f(0.f, 0.f, 1.f);
            glVertex3f(0.f, 0.6f, 0.f);
            glEnd();
            glfwSwapBuffers(window);
            glfwPollEvents();
        }

    } else {
        printf("Unsuccessful!\n");
    }
}
