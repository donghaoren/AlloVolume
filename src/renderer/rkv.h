// Compute and output.
template <typename real_t, typename output_t, typename dxdt_t, typename norm_func_t>
__device__ inline bool RungeKuttaVerner(
    real_t t0, real_t t1,                                // t0, t1
    output_t x0, dxdt_t& dxdt, norm_func_t& norm_func,     // x0, dxdt(x, t)
    real_t TOL, real_t hmin, real_t hmax,                // step size control, and error tolerance
    output_t& x1                                         // output
) {
    bool ret = true;
    real_t t = t0;
    real_t h = hmax;
    if(t + h > t1) h = t1 - t;

    output_t w, k1, k2, k3, k4, k5, k6, k7, k8;
    output_t dx;
    output_t tmp;
    output_t w_new, w_new_hat;

    w = x0;

    bool continue_evaluation = true;
    bool last_step = false;

    while(continue_evaluation) {
        // First compute k1 ~ k8.
        dxdt(t, w, dx);
        k1 = dx * h;
        tmp = w + k1 / 6.0;
        dxdt(t + h / 6.0, tmp, dx);
        k2 = dx * h;
        tmp = w + k1 * (4.0/75.0) + k2 * (16.0/75.0);
        dxdt(t + h * (4.0/15.0), tmp, dx);
        k3 = dx * h;
        tmp = w + k1 * (5.0/6.0) - k2 * (8.0/3.0) + k3 * (5.0/2.0);
        dxdt(t + h * (2.0/3.0), tmp, dx);
        k4 = dx * h;
        tmp = w - k1 * (165.0/64.0) + k2 * (55.0/6.0) - k3 * (425.0/64.0) + k4 * (85.0/96.0);
        dxdt(t + h * (5.0/6.0), tmp, dx);
        k5 = dx * h;
        tmp = w + k1 * (12.0/5.0) - k2 * 8.0 + k3 * (4015.0/612.0) - k4 * (11.0/36.0) + k5 * (88.0/255.0);
        dxdt(t + h, tmp, dx);
        k6 = dx * h;
        tmp = w - k1 * (8263.0/15000.0) + k2 * (124.0/75.0) - k3 * (643.0/680.0) - k4 * (81.0/250.0) + k5 * (2484.0/10625.0);
        dxdt(t + h / 15.0, tmp, dx);
        k7 = dx * h;
        tmp = w + k1 * (3501.0/1720.0) - k2 * (300.0/43.0) + k3 * (297275.0/52632.0) - k4 * (319.0/2322.0) + k5 * (24068.0/84065.0) + k7 * (3850.0/26703.0);
        dxdt(t + h, tmp, dx);
        k8 = dx * h;
        w_new = w + k1 * (13.0/160.0) + k3 * (2375.0/5984.0) + k4 * (5.0/16.0) + k5 * (12.0/85.0) + k6 * (3.0/44.0);
        w_new_hat = w + k1 * (3.0/40.0) + k3 * (875.0/2244.0) + k4 * (23.0/72.0) + k5 * (264.0/1955.0) + k7 * (125.0/11592.0) + k8 * (43.0/616.0);
        real_t R = norm_func(w_new - w_new_hat);
        if(R <= TOL) {
            t = t + h;
            w = w_new;
        }
        real_t delta = 0.84 * pow(TOL / R, 1.0/5.0); // fifth-order method.
        if(delta < 0.1) {
            h = 0.1 * h;
        } else if(delta > 4.0) {
            h = 4.0 * h;
        } else h = delta * h;
        if(h > hmax) h = hmax;
        if(t >= t1 || last_step) {
            continue_evaluation = false;
        } else if(t + h > t1) {
            h = t1 - t;
            last_step = true;
        } else if(h < hmin) {
            h = hmin;
        }
    }

    // We're at t1.
    x1 = w;
    return ret;
}


// Compute and output.
template <typename real_t, typename output_t, typename dxdt_t, typename norm_func_t>
__device__ inline bool RungeKuttaFehlberg(
    real_t t0, real_t t1,                                // t0, t1
    output_t x0, dxdt_t& dxdt, norm_func_t& norm_func,     // x0, dxdt(x, t)
    real_t TOL, real_t hmin, real_t hmax,                // step size control, and error tolerance
    output_t& x1                                         // output
) {
    bool ret = true;
    real_t t = t0;
    real_t h = hmax;
    if(t + h > t1) h = t1 - t;

    output_t w, k1, k2, k3, k4, k5, k6;
    output_t dx;
    output_t w_new, w_new_hat;

    w = x0;

    bool force_step = false;

    for(;;) {
        // First compute k1 ~ k8.
        dxdt(t, w, dx);
        k1 = dx * h;
        dxdt(t + h / 4.0f, w + k1 / 4.0f, dx);
        k2 = dx * h;
        dxdt(t + h * (3.0f/8.0f), w + k1 * (3.0f/32.0f) + k2 * (9.0f/32.0f), dx);
        k3 = dx * h;
        dxdt(t + h * (12.0f/13.0f), w + k1 * (1932.0f/2197.0f) - k2 * (7200.0f/2197.0f) + k3 * (7296.0f/2197.0f), dx);
        k4 = dx * h;
        dxdt(t + h, w + k1 * (439.0f/216.0f) - k2 * 8.0f + k3 * (3680.0f/513.0f) - k4 * (845.0f/4104.0f), dx);
        k5 = dx * h;
        dxdt(t + h / 2.0f, w - k1 * (8.0f/27.0f) + k2 * 2.0f - k3 * (3544.0f/2565.0f) + k4 * (1859.0f/4104.0f) - k5 * (11.0f/40.0f), dx);
        k6 = dx * h;
        real_t R = norm_func(k1 / 360.0f - k3 * (128.0f/4275.0f) - k4 * (2197.0f/75240.0f) + k5 / 50.0f + k6 * (2.0f/55.0f)) / h;
        if(R <= TOL || force_step) {
            t = t + h;
            w = w + k1 * (25.0f/216.0f) + k3 * (1408.0f/2565.0f) + k4 * (2197.0f/4104.0f) - k5 / 5.0f;
            force_step = false;
        }
        real_t delta = 0.84f * powf(TOL / R, 1.0f/4.0f); // fourth-order method.
        if(delta < 0.1f) {
            h = 0.1f * h;
        } else if(delta > 4.0f) {
            h = 4.0f * h;
        } else h = delta * h;

        if(h > hmax) h = hmax;
        if(h < hmin) {
            h = hmin;
            force_step = true;
        }
        if(t >= t1) {
            break;
        } else if(t + h > t1) {
            h = t1 - t;
        }
    }

    // We're at t1.
    x1 = w;
    return ret;
}
