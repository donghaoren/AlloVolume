#include <stdio.h>
#include <stdlib.h>

#ifdef __APPLE__
    #include <GLUT/glut.h>
#else
    #include <GL/glut.h>
#endif

void display_callback() {
    int width = glutGet(GLUT_WINDOW_WIDTH);
    int height = glutGet(GLUT_WINDOW_HEIGHT);
    printf("Display: %d %d\n", width, height);
    glutSwapBuffers();
}

void keyboard_callback(unsigned char key, int x, int y) {
    exit(0);
}

int main(int argc, char* argv[]) {
    glutInit(&argc, argv);

    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_STEREO);

    bool game_mode = (argc == 1);

    if(game_mode) {
        int width = glutGet(GLUT_SCREEN_WIDTH);
        int height = glutGet(GLUT_SCREEN_HEIGHT);
        printf("%d %d\n", width, height);
        char game_mode_string[64];
        sprintf(game_mode_string, "%dx%d:24", width, height);
        glutGameModeString(game_mode_string);
        glutEnterGameMode();
        glutDisplayFunc(display_callback);
        glutKeyboardFunc(keyboard_callback);
    } else {
        glutCreateWindow("AlloVolume");
        glutDisplayFunc(display_callback);
        glutKeyboardFunc(keyboard_callback);
    }

    glutMainLoop();
}
