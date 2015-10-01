// Experiment with Ray Traversal algorithm.
// See: A Fast Voxel Traversal Algorithm for Ray Tracing
// John Amanatides, Andrew Woo

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

struct Vector {
    float x, y, z;
};

struct Ray {
    Vector p, d;
};

unsigned int interleave_X(int X) { return X; }
unsigned int interleave_Y(int Y) { return Y; }
unsigned int interleave_Z(int Z) { return Z; }
float get_voxel(unsigned int X, unsigned int Y, unsigned int Z) { return X + Y + Z; }

// p, d in volume coordinates, p must be inside the volume.
void volume_traversal(Vector p, Vector d, int tmax) {
    // Current voxel (box).
    int X = p.x, Y = p.y, Z = p.z;

    // Stepping control.
    int step_x, step_y, step_z;
    float t_adv_x, t_adv_y, t_adv_z;
    float t_max_x, t_max_y, t_max_z;

    // Shortcuts.
    unsigned int Xi0 = interleave_X(X);
    unsigned int Yi0 = interleave_Y(Y);
    unsigned int Zi0 = interleave_Z(Z);
    unsigned int Xi1 = interleave_X(X + 1);
    unsigned int Yi1 = interleave_Y(Y + 1);
    unsigned int Zi1 = interleave_Z(Z + 1);

    float v000 = get_voxel(Xi0, Yi0, Zi0);
    float v001 = get_voxel(Xi0, Yi0, Zi1);
    float v010 = get_voxel(Xi0, Yi1, Zi0);
    float v011 = get_voxel(Xi0, Yi1, Zi1);
    float v100 = get_voxel(Xi1, Yi0, Zi0);
    float v101 = get_voxel(Xi1, Yi0, Zi1);
    float v110 = get_voxel(Xi1, Yi1, Zi0);
    float v111 = get_voxel(Xi1, Yi1, Zi1);


    float t = 0;

    // Initialize t_max_x, t_max_y, t_max_z.
    if(d.x > 0) {
        t_max_x = (X + 1 - p.x) / d.x; step_x = 1; t_adv_x = 1.0 / d.x;
    } else if(d.x < 0) {
        t_max_x = (X - p.x) / d.x; t_adv_x = -1.0 / d.x; step_x = -1;
    } else {
        t_max_x = 1e20; t_adv_x = 0; step_x = 0;
    }

    if(d.y > 0) {
        t_max_y = (Y + 1 - p.y) / d.y; step_y = 1; t_adv_y = 1.0 / d.y;
    } else if(d.y < 0) {
        t_max_y = (Y - p.y) / d.y; t_adv_y = -1.0 / d.y; step_y = -1;
    } else {
        t_max_y = 1e20; t_adv_y = 0; step_y = 0;
    }

    if(d.z > 0) {
        t_max_z = (Z + 1 - p.z) / d.z; step_z = 1; t_adv_z = 1.0 / d.z;
    } else if(d.z < 0) {
        t_max_z = (Z - p.z) / d.z; t_adv_z = -1.0 / d.z; step_z = -1;
    } else {
        t_max_z = 1e20; t_adv_z = 0; step_z = 0;
    }

    while(t < tmax) {
        float t1 = fmin(tmax, fmin(t_max_z, fmin(t_max_x, t_max_y)));
        if(t1 > t) {
            // Do voxel.
            printf("[%d, %d, %d, %f, %f],\n", X, Y, Z, t, t1);
        }

        if(t_max_x < t_max_y) {
            if(t_max_x < t_max_z) {
                t = t_max_x;
                t_max_x = t_max_x + t_adv_x;
                if(step_x > 0) {
                    X += 1;
                    v000 = v100;
                    v001 = v101;
                    v010 = v110;
                    v011 = v111;
                    Xi0 = Xi1;
                    Xi1 = interleave_X(X + 1);
                    v100 = get_voxel(Xi1, Yi0, Zi0);
                    v101 = get_voxel(Xi1, Yi0, Zi1);
                    v110 = get_voxel(Xi1, Yi1, Zi0);
                    v111 = get_voxel(Xi1, Yi1, Zi1);
                } else {
                    X -= 1;
                    v100 = v000;
                    v101 = v001;
                    v110 = v010;
                    v111 = v011;
                    Xi1 = Xi0;
                    Xi0 = interleave_X(X);
                    v000 = get_voxel(Xi0, Yi0, Zi0);
                    v001 = get_voxel(Xi0, Yi0, Zi1);
                    v010 = get_voxel(Xi0, Yi1, Zi0);
                    v011 = get_voxel(Xi0, Yi1, Zi1);
                }
            } else {
                t = t_max_z;
                t_max_z = t_max_z + t_adv_z;
                if(step_z > 0) {
                    Z += 1;
                    v000 = v001;
                    v010 = v011;
                    v100 = v101;
                    v110 = v111;
                    Zi0 = Zi1;
                    Zi1 = interleave_Z(Z + 1);
                    v001 = get_voxel(Xi0, Yi0, Zi1);
                    v011 = get_voxel(Xi0, Yi1, Zi1);
                    v101 = get_voxel(Xi1, Yi0, Zi1);
                    v111 = get_voxel(Xi1, Yi1, Zi1);
                } else {
                    Z -= 1;
                    v001 = v000;
                    v011 = v010;
                    v101 = v100;
                    v111 = v110;
                    Zi1 = Zi0;
                    Zi0 = interleave_Z(Z);
                    v000 = get_voxel(Xi0, Yi0, Zi0);
                    v010 = get_voxel(Xi0, Yi1, Zi0);
                    v100 = get_voxel(Xi1, Yi0, Zi0);
                    v110 = get_voxel(Xi1, Yi1, Zi0);
                }
            }
        } else {
            if(t_max_y < t_max_z) {
                t = t_max_y;
                t_max_y = t_max_y + t_adv_y;
                if(step_y > 0) {
                    Y += 1;
                    v000 = v010;
                    v001 = v011;
                    v100 = v110;
                    v101 = v111;
                    Yi0 = Yi1;
                    Yi1 = interleave_Y(Y + 1);
                    v010 = get_voxel(Xi0, Yi1, Zi0);
                    v011 = get_voxel(Xi0, Yi1, Zi1);
                    v110 = get_voxel(Xi1, Yi1, Zi0);
                    v111 = get_voxel(Xi1, Yi1, Zi1);
                } else {
                    Y -= 1;
                    v010 = v000;
                    v011 = v001;
                    v110 = v100;
                    v111 = v101;
                    Yi1 = Yi0;
                    Yi0 = interleave_Y(Y);
                    v000 = get_voxel(Xi0, Yi0, Zi0);
                    v001 = get_voxel(Xi0, Yi0, Zi1);
                    v100 = get_voxel(Xi1, Yi0, Zi0);
                    v101 = get_voxel(Xi1, Yi0, Zi1);
                }
            } else {
                t = t_max_z;
                t_max_z = t_max_z + t_adv_z;
                if(step_z > 0) {
                    Z += 1;
                    v000 = v001;
                    v010 = v011;
                    v100 = v101;
                    v110 = v111;
                    Zi0 = Zi1;
                    Zi1 = interleave_Z(Z + 1);
                    v001 = get_voxel(Xi0, Yi0, Zi1);
                    v011 = get_voxel(Xi0, Yi1, Zi1);
                    v101 = get_voxel(Xi1, Yi0, Zi1);
                    v111 = get_voxel(Xi1, Yi1, Zi1);
                } else {
                    Z -= 1;
                    v001 = v000;
                    v011 = v010;
                    v101 = v100;
                    v111 = v110;
                    Zi1 = Zi0;
                    Zi0 = interleave_Z(Z);
                    v000 = get_voxel(Xi0, Yi0, Zi0);
                    v010 = get_voxel(Xi0, Yi1, Zi0);
                    v100 = get_voxel(Xi1, Yi0, Zi0);
                    v110 = get_voxel(Xi1, Yi1, Zi0);
                }
            }
        }
    }
}

float randdirection() {
    return (rand() % 20 - 10) / 10.0;
}

int main() {
    srand(time(0));
    printf("[");
    Vector p = { rand() % 10 / 7.0 + 10, rand() % 10 / 7.0 + 10, rand() % 10 / 7.0 + 10 };
    Vector d = { randdirection(), randdirection(), 0 };
    // Vector p = { 10.5, 10.8, 0 };
    // Vector d = { 1, -1, 0 };
    volume_traversal(p, d, 5);
    printf("[%f,%f,%f,%f,%f,%f]]\n", p.x, p.y, p.z, d.x, d.y, d.z);
}
