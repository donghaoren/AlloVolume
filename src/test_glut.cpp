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
    if(key == 'a') {
        glutEnterGameMode();
        glutDisplayFunc(display_callback);
        glutKeyboardFunc(keyboard_callback);
    } else if(key == 'b') {
        glutLeaveGameMode();
        glutDisplayFunc(display_callback);
        glutKeyboardFunc(keyboard_callback);
    } else {
        exit(0);
    }
}

int main(int argc, char* argv[]) {
    glutInit(&argc, argv);

    int width = glutGet(GLUT_SCREEN_WIDTH);
    int height = glutGet(GLUT_SCREEN_HEIGHT);

    printf("%d %d\n", width, height);

    char game_mode_string[64];
    sprintf(game_mode_string, "%dx%d:24", width, height);

    glutGameModeString(game_mode_string);

    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);

    glutCreateWindow("AlloVolume");

    glutDisplayFunc(display_callback);
    glutKeyboardFunc(keyboard_callback);

    glutMainLoop();
}
