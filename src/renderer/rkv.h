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
    bool is_first = true;

    while(continue_evaluation) {
        // First compute k1 ~ k8.
        dxdt(t, w, dx);
        k1 = dx * h;
        tmp = w + k1 / 6.0f;
        dxdt(t + h / 6, tmp, dx);
        k2 = dx * h;
        tmp = w + k1 * (4.0/75.0) + k2 * (16.0/75.0);
        dxdt(t + h * (4.0/15.0), tmp, dx);
        k3 = dx * h;
        tmp = w + k1 * (5.0/6.0) - k2 * (8.0/3.0) + k3 * (5.0/2.0);
        dxdt(t + h * (2.0/3.0), tmp, dx);
        k4 = dx * h;
        tmp = w - k1 * (165.0/64.0) + k2 * (55.0/6.0) - k3 * (425.0/64.0) + k4 * (85.0/96.0);
        dxdt(t + h * (5.0 / 6.0), tmp, dx);
        k5 = dx * h;
        tmp = w + k1 * (12.0/5.0) - k2 * 8 + k3 * (4015.0/612.0) - k4 * (11.0/36.0) + k5 * (88.0/255.0);
        dxdt(t + h, tmp, dx);
        k6 = dx * h;
        tmp = w - k1 * (8263.0/15000.0) + k2 * (124.0/75.0) - k3 * (643.0/680.0) - k4 * (81.0/250.0) + k5 * (2484.0/10625.0);
        dxdt(t + h / 15.0, tmp, dx);
        k7 = dx * h;
        tmp = w + k1 * (3501.0/1720.0) - k2 * (300.0/43.0) + k3 * (297275.0/52632.0) - k4 * (319.0/2322.0) + k5 * (24068.0/84065.0) + k7 * (3850.0/26703.0);
        dxdt(t+h,tmp,dx);
        k8 = dx * h;
        w_new = w + k1 * (13.0/160.0) + k3 * (2375.0/5984.0) + k4 * (5.0/16.0) + k5 * (12.0/85.0) + k6 * (3.0/44.0);
        w_new_hat = w + k1 * (3.0/40.0) + k3 * (875.0/2244.0) + k4 * (23.0/72.0) + k5 * (264.0/1955.0) + k7 * (125.0/11592.0) + k8 * (43.0/616.0);
        real_t R = norm_func(w_new - w_new_hat);
        if(R <= TOL) {
            t = t + h;
            w = w_new;
        }
        real_t delta = 0.84 * pow(TOL / R, 1.0f/5.0f); // fifth-order method.
        if(delta < 0.1) {
            h = 0.1 * h;
        } else if(delta > 4) {
            h = 4 * h;
        } else h = delta * h;
        if(h > hmax) h = hmax;
        if(t >= t1) {
            continue_evaluation = 0;
        } else if(t + h > t1) {
            h = t1 - t;
        } else if(h < hmin) {
            continue_evaluation = 0;
            // printf("RKVError: minimum h reached, but still can't make error within TOL.\n");
            ret = false;
        }
    }

    // We're at t1.
    x1 = w;
    return ret;
}

// This needs more debugging...

// template <typename real_t, typename output_t, typename dxdt_t, typename norm_func_t>
// __device__ inline bool RungeKuttaCashKarp(
//     real_t t0, real_t t1,                                // t0, t1
//     output_t x0, dxdt_t& dxdt, norm_func_t& norm_func,     // x0, dxdt(x, t)
//     real_t TOL, real_t hmin, real_t hmax,                // step size control, and error tolerance
//     output_t& x1                                         // output
// ) {
//     bool ret = true;
//     real_t t = t0;
//     real_t h = hmax;
//     if(t + h > t1) h = t1 - t;

//     output_t w;
//     output_t tmp;
//     output_t dydx;
//     output_t ak2, ak3, ak4, ak5, ak6;

//     w = x0;

//     bool continue_evaluation = true;
//     bool is_first = true;

//     while(continue_evaluation) {
//         const float
//             a2 = 0.2, a3 = 0.3, a4 = 0.6, a5 = 1.0, a6 = 0.875,
//             b21 = 0.2,
//             b31 = 3.0 / 40.0, b32 = 9.0 / 40.0,
//             b41 = 0.3, b42 = -0.9, b43 = 1.2,
//             b51 = -11.0 / 54.0, b52 = 2.5, b53 = -70.0 / 27.0, b54 = 35.0 / 27.0,
//             b61 = 1631.0 / 55296.0, b62 = 175.0 / 512.0, b63 = 575.0 / 13824.0, b64 = 44275.0 / 110592.0, b65 = 253.0 / 4096.0,
//             c1 = 37.0 / 378.0, c3 = 250.0 / 621.0, c4 = 125.0 / 594.0, c6 = 512.0 / 1771.0,
//             dc5 = -277.00 / 14336.0;
//         const float
//             dc1 = c1 - 2825.0 / 27648.0,
//             dc3 = c3 - 18575.0 / 48384.0,
//             dc4 = c4 - 13525.0 / 55296.0,
//             dc6 = c6 - 0.25;

//         dxdt(t, w, dydx);
//         tmp = w + dydx * (b21 * h);
//         dxdt(t + a2 * h, tmp, ak2); // Second step.
//         tmp = w + (dydx * b31 + ak2 * b32) * h;
//         dxdt(t + a3 * h, tmp, ak3); // Third step.
//         tmp = w + (dydx * b41 + ak2 * b42 + ak3 * b43) * h;
//         dxdt(t + a4 * h, tmp, ak4); // Fourth step.
//         tmp = w + (dydx * b51 + ak2 * b52 + ak3 * b53 + ak4 * b54) * h;
//         dxdt(t + a5 * h, tmp, ak5); // Fifth step.
//         tmp = w + (dydx * b61 + ak2 * b62 + ak3 * b63 + ak4 * b64 + ak5 * b65) * h;
//         dxdt(t + a6 * h, tmp, ak6); // Sixth step.
//         // Accumulate increments with proper weights.
//         output_t wout = w + (dydx * c1 + ak3 * c3 + ak4 * c4 + ak6 * c6) * h;
//         // Estimate error as difference between fourth and fifth order methods.
//         output_t werr = (dydx * dc1 + ak3 * dc3 + ak4 * dc4 + ak5 * dc5 + ak6 * dc6) * h;
//         real_t R = norm_func(werr);
//         if(R <= TOL) {
//             t = t + h;
//             w = wout;
//         }
//         real_t delta = 0.84 * pow(TOL / R, 1.0f/5.0f); // fifth-order method.
//         if(delta < 0.1) {
//             h = 0.1 * h;
//         } else if(delta > 4) {
//             h = 4 * h;
//         } else h = delta * h;
//         if(h > hmax) h = hmax;
//         if(t >= t1) {
//             continue_evaluation = 0;
//         } else if(t + h > t1) {
//             h = t1 - t;
//         } else if(h < hmin) {
//             continue_evaluation = 0;
//             // printf("RKVError: minimum h reached, but still can't make error within TOL.\n");
//             ret = false;
//         }
//     }

//     // We're at t1.
//     x1 = w;
//     return ret;
// }
