var Gradients = {
    "rainbow": { name: "Rainbow", values: [
        [ 0.471412, 0.108766, 0.527016 ],
        [ 0.274861, 0.182264, 0.727279 ],
        [ 0.248488, 0.386326, 0.813373 ],
        [ 0.297960, 0.565793, 0.752239 ],
        [ 0.388225, 0.674195, 0.603544 ],
        [ 0.513417, 0.729920, 0.440682 ],
        [ 0.666083, 0.743042, 0.322935 ],
        [ 0.808342, 0.711081, 0.255976 ],
        [ 0.893514, 0.600415, 0.220546 ],
        [ 0.892955, 0.389662, 0.179401 ],
        [ 0.857359, 0.131106, 0.132128 ]
    ] },
    "temperature": { name: "Temperature", values: [
        [ 0.178927, 0.305394, 0.933501 ],
        [ 0.337660, 0.466886, 0.942736 ],
        [ 0.528934, 0.628452, 0.956059 ],
        [ 0.748934, 0.803792, 0.975610 ],
        [ 0.912556, 0.933111, 0.991522 ],
        [ 0.984192, 0.987731, 0.911643 ],
        [ 0.994726, 0.991128, 0.667358 ],
        [ 0.977887, 0.937070, 0.368596 ],
        [ 0.924921, 0.740045, 0.258448 ],
        [ 0.867569, 0.491545, 0.211209 ],
        [ 0.817319, 0.134127, 0.164218 ]
    ] }
};

var Colors = [
  "000000", "ffffff", "888888" ,"1f77b4", "ff7f0e", "2ca02c", "d62728", "9467bd", "8c564b", "e377c2", "bcbd22", "17becf",
  "BD0026", "F03B20", "FD8D3C" ,"FEB24C", "FED976", "FFFFB2", "045A8D", "2B8CBE", "74A9CF", "A6BDDB", "D0D1E6", "F1EEF6",
  "0A4404", "2B681A", "528D38" ,"7FB35C", "B3D787", "EEFBBA", "4F0555", "7C257E", "A64BA1", "CB77BB", "E9A8CB", "FDDDD0",
  "252525", "636363", "969696" ,"BDBDBD", "D9D9D9", "F7F7F7", "B2182B", "EF8A62", "FDDBC7", "D1E5F0", "67A9CF", "2166AC"
].map(function(str) {
    var r = eval("0x" + str.substr(0, 2));
    var g = eval("0x" + str.substr(2, 2));
    var b = eval("0x" + str.substr(4, 2));
    return [ r / 255, g / 255, b / 255 ];
});

var clamp = function(val, min, max) {
    return val < min ? min : (val > max ? max : val);
};

var sample_gradient = function(gradient, t) {
    var pos = t * gradient.length;
    var i = parseInt(Math.floor(pos));
    if(i < 0) i = 0;
    if(i >= gradient.length - 1) i = gradient.length - 2;
    var k = pos - i;
    return {
        r: gradient[i][0] * (1 - k) + gradient[i + 1][0] * k,
        g: gradient[i][1] * (1 - k) + gradient[i + 1][1] * k,
        b: gradient[i][2] * (1 - k) + gradient[i + 1][2] * k,
        a: 1.0
    };
};

var array_to_color = function(a) {
    return { r: a[0], g: a[1], b: a[2], a: a[3] === undefined ? 1.0 : a[3] };
};

var rgba_color = function(c, alpha) {
    var r = Math.max(0, Math.min(255, parseInt(c.r * 255)));
    var g = Math.max(0, Math.min(255, parseInt(c.g * 255)));
    var b = Math.max(0, Math.min(255, parseInt(c.b * 255)));
    var a = Math.max(0, Math.min(1, c.a.toFixed(3)));
    if(alpha !== undefined) a = alpha;
    return "rgba(" + r + "," + g + "," + b + "," + a + ")";
};

var blend_color = function(src, dest) {
    var result = {
        r: dest.r * (1 - src.a) * dest.a + src.r * src.a,
        g: dest.g * (1 - src.a) * dest.a + src.g * src.a,
        b: dest.b * (1 - src.a) * dest.a + src.b * src.a,
        a: dest.a * (1 - src.a) + src.a
    };
    if(result.a != 0) {
        result.r /= result.a;
        result.g /= result.a;
        result.b /= result.a;
        return result;
    } else {
        return { r: 0, g: 0, b: 0, a: 0 };
    }
};
