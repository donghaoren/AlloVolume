// This header includes OpenGL.
#ifndef ALLOVOLUME_OPENGL_INCLUDE_H
#define ALLOVOLUME_OPENGL_INCLUDE_H

#ifdef __APPLE__
    #include <OpenGL/gl.h>
    #include <OpenGL/glext.h>
    #include <GLUT/glut.h>
    // Some missing defines.
    #define GL_RGBA32F GL_RGBA32F_ARB
    #define GL_RGB32F GL_RGB32F_ARB
#else
    #include <GL/glew.h>
    #include <GL/glut.h>
    #include <GL/freeglut_ext.h>
#endif

#endif
