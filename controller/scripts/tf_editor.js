// Transfer function editor.

var TransferFunctionDescription = function() {
    this.domain = [ 1e-3, 1e8 ];
    this.is_log = true;
    this.gradient_stops = [ ];
    this.gradient_alpha_power = 1;
    this.gradient_alpha_max = 0.1;
    this.gaussians = [ ];
};

TransferFunctionDescription.Gradients = {
    "rainbow": { name: "Rainbow", type: "uniform", values: [
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
    "temperature": { name: "Temperature", type: "uniform", values: [
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

var blend_color_old = function(src, dest) {
    var result = {
        r: dest.r * (1 - src.a) + src.r * src.a,
        g: dest.g * (1 - src.a) + src.g * src.a,
        b: dest.b * (1 - src.a) + src.b * src.a,
        a: dest.a * (1 - src.a) + src.a * src.a
    };
    return result;
};

TransferFunctionDescription.prototype.scale = function(value) {
    var td = this.domain;
    var tv = value;
    if(this.is_log) {
        td = [ Math.log(td[0]), Math.log(td[1]) ];
        tv = Math.log(tv);
    }
    return (tv - td[0]) / (td[1] - td[0]);
};

TransferFunctionDescription.prototype.inverse = function(value) {
    var td = this.domain;
    if(this.is_log) {
        td = [ Math.log(td[0]), Math.log(td[1]) ];
    }
    var tv = value * (td[1] - td[0]) + td[0];
    if(this.is_log) {
        tv = Math.exp(tv);
    }
    return tv;
};

TransferFunctionDescription.prototype.sampleGradient = function(t) {
    for(var i = 0; i < this.gradient_stops.length - 1; i++) {
        var t0 = this.gradient_stops[i].t;
        var t1 = this.gradient_stops[i + 1].t;
        if(t0 == t1) continue;
        if(t0 <= t && t <= t1) {
            var p = (t - t0) / (t1 - t0);
            var c1 = this.gradient_stops[i], c2 = this.gradient_stops[i + 1];
            var alpha = this.gradient_alpha_max * Math.pow(t, this.gradient_alpha_power);
            return {
                r: c1.r * (1 - p) + c2.r * p,
                g: c1.g * (1 - p) + c2.g * p,
                b: c1.b * (1 - p) + c2.b * p,
                a: (c1.a * (1 - p) + c2.a * p) * alpha,
            };
        }
    }
    return {
        r: 0, g: 0, b: 0, a: 0
    };
};

TransferFunctionDescription.prototype.sampleGaussian = function(t) {
    var c = null;
    for(var i = 0; i < this.gaussians.length; i++) {
        var g = this.gaussians[i];
        var a = g.color.a * Math.exp(-Math.pow((g.center - t) / g.sigma, 2) / 2);
        var color = { r: g.color.r, g: g.color.g, b: g.color.b, a: a };
        if(c) c = blend_color(color, c);
        else c = color;
    }
    return c;
};

TransferFunctionDescription.prototype.sample = function(t) {
    var g1 = this.sampleGradient(t);
    var g2 = this.sampleGaussian(t);
    if(g2) {
        return blend_color(g2, g1);
    } else {
        return g1;
    }
};

TransferFunctionDescription.prototype.generateTexture = function(size) {
    if(!size) size = 1600;
    var samples = [];
    for(var i = 0; i < size; i++) {
        var t = (i + 0.5) / size;
        var r = this.sample(t);
        samples.push([ r.r, r.g, r.b, r.a ]);
    }
    return samples;
};

TransferFunctionDescription.prototype.generateGradient = function(gradient, alpha_max, alpha_pow) {
    if(alpha_max === undefined) alpha_max = 0.1;
    if(alpha_pow === undefined) alpha_pow = 1;
    this.gradient_alpha_max = alpha_max;
    this.gradient_alpha_power = alpha_pow;
    var self = this;
    if(typeof(gradient) == "string") {
        gradient = TransferFunctionDescription.Gradients[gradient];
    }
    if(gradient.type == "uniform") {
        var length = gradient.values.length;
        this.gradient_stops = [];
        for(var index = 0; index < gradient.values.length; index++) {
            var rgb = gradient.values[index];
            var t = index / (length - 1);
            this.gradient_stops.push({ t: t, r: rgb[0], g: rgb[1], b: rgb[2], a: rgb.length == 4 ? rgb[3] : 1 });
        }
    }
};

TransferFunctionDescription.prototype.generateGaussians = function(tmin, tmax, count) {
    for(var i = 1; i <= count; i++) {
        var t = i / (count) * (tmax - tmin) + tmin;
        var c = this.sampleGradient(t);
        c.a = this.sampleGradient(t).a;
        this.gaussians.push({
            center: t,
            sigma: (tmax - tmin) / (count) / 10,
            color: c
        });
    }
};

var TransferFunctionEditor = function(wrapper) {
    var self = this;
    this.wrapper = wrapper;

    this.container = wrapper.append("div")
      .style("display", "block")
      .style("background-color", "rgba(0, 0, 0, 1)");

    this.graphics = this.container.append("div").attr("class", "graphics");
    this.controls = this.container.append("div").attr("class", "controls");
    this.canvas = this.graphics.append("canvas");
    this.svg = this.graphics.append("svg");

    this.tf = new TransferFunctionDescription();
    this.tf.generateGradient("rainbow", 1, 1);
    this.tf.generateGaussians(0, 1, 10);
    this.tf.generateGradient("rainbow", 0, 1);

    this.svg.append("g").attr("class", "axis");
    this.svg.append("g").attr("class", "axis_upper axis");
    this.svg.append("g").attr("class", "axis_alpha axis");
    this.svg.append("g").attr("class", "gradient_stops");
    this.svg.append("g").attr("class", "gaussians");
    this.svg.append("path").attr("class", "gradient_max");
    this.svg.append("path").attr("class", "gradient_power");

    this.buttons = this.controls.append("div");

    this.range_min_input = this.buttons.append("input").attr("type", "text");
    this.range_max_input = this.buttons.append("input").attr("type", "text");
    this.buttons.append("span")
      .attr("class", "btn")
      .text("Range")
      .on("click", function() {
        self.tf.domain = [ parseFloat(self.range_min_input.node().value), parseFloat(self.range_max_input.node().value) ];
        self.render();
      });

    this.gradient_select = this.buttons.append("select");
    var gradients = [];
    for(var key in TransferFunctionDescription.Gradients) {
        var g = TransferFunctionDescription.Gradients[key];
        gradients.push({
            name: g.name,
            key: key
        });
    }
    this.gradient_select.selectAll("option").data(gradients).enter()
      .append("option")
        .attr("value", function(d) { return d.key; })
        .text(function(d) { return d.name; });

    this.gradient_alpha = this.buttons.append("input").attr("type", "text").attr("value", 1);
    this.gradient_power = this.buttons.append("input").attr("type", "text").attr("value", 1);

    this.buttons.append("span")
      .attr("class", "btn")
      .text("Gradient")
      .on("click", function() {
        var alpha = parseFloat(self.gradient_alpha.node().value);
        var power = parseFloat(self.gradient_power.node().value);
        self.tf.generateGradient(self.gradient_select.node().value, alpha, power);
        self.render();
      });

    this.gaussians_input = this.buttons.append("input").attr("type", "text").attr("value", 10);

    this.buttons.append("span")
      .attr("class", "btn")
      .text("Gaussians")
      .on("click", function() {
        var count = parseInt(self.gaussians_input.node().value);
        self.tf.gaussians = [];
        if(count > 0) {
            self.tf.generateGaussians(0, 1, count);
        }
        self.render();
      });

    this.canvas_height = 200;
    this.resize();
};

TransferFunctionEditor.prototype.resize = function() {
    var width = parseInt(this.wrapper.style("width"));
    this.container
      .style("width", width + "px");
    this.graphics
      .style("width", width + "px")
      .style("height", this.canvas_height + "px");
    this.canvas.node().width = width * 2;
    this.canvas.node().height = this.canvas_height * 2;
    this.canvas
      .style("width", width + "px")
      .style("height", this.canvas_height + "px");
    this.svg.style("width", width + "px");
    this.svg.style("height", this.canvas_height + "px");

    this.width = width;

    this.render();
};

var glyph_droplet = function(height, span, radius) {
    if(radius === undefined) {
        var s = span / 2, h = height;
        radius = Math.sqrt(s * s * s * s / h / h + s * s);
    }
    return ["M","0","0","L", span / 2, height, "A", radius, radius,"0","1","1", -span / 2, height,"Z"].join(" ");
};

TransferFunctionEditor.prototype.render = function() {
    var self = this;
    var tf = this.tf;

    var number_format = d3.format("e");
    self.range_min_input.node().value = number_format(tf.domain[0]);
    self.range_max_input.node().value = number_format(tf.domain[1]);
    self.gradient_alpha.node().value = tf.gradient_alpha_max.toFixed(3);
    self.gradient_power.node().value = tf.gradient_alpha_power.toFixed(3);


    var axis_y = this.canvas_height - 40;
    var axis_y_upper = axis_y - 26;

    var margin = 40;
    var width = this.width - margin * 2;

    var scale = d3.scale.linear();
    if(tf.is_log) {
        scale = d3.scale.log().base(10);
    }

    scale
      .domain(tf.domain)
      .range([ margin, width + margin ]);

    var alpha_scale = d3.scale.linear();
    alpha_scale
      .domain([0, 1])
      .range([ axis_y_upper, 20 ]);

    var axis = d3.svg.axis()
      .scale(scale)
      .orient("top");

    var axis_upper = d3.svg.axis()
      .scale(scale)
      .orient("bottom");

    var axis_alpha = d3.svg.axis()
      .scale(alpha_scale)
      .ticks(5)
      .orient("left");

    this.svg.select("g.axis").attr("transform", "translate(0, " + axis_y + ")").call(axis);
    this.svg.select("g.axis_upper").attr("transform", "translate(0, " + axis_y_upper + ")").call(axis_upper);
    this.svg.select("g.axis_alpha").attr("transform", "translate(40, 0)").call(axis_alpha);

    // var path_gradients = this.svg.select("g.gradient_stops").selectAll("g").data(tf.gradient_stops);
    // path_gradients.enter().append("g").append("path");
    // var drag_gradients = d3.behavior.drag();
    // path_gradients
    //   .attr("transform", function(data) {
    //       var pos = scale(tf.inverse(data.t));
    //       return "translate(" + pos + "," + (axis_y + 6) + ")";
    //   })
    //   .call(drag_gradients);
    // path_gradients.select("path")
    //   .attr("fill", function(data) { return rgba_color(data, 1); })
    //   .attr("d", glyph_droplet(10, 14));
    // drag_gradients.on("drag", function(data) {
    //     data.t = Math.min(1, Math.max(0, tf.scale(scale.invert(d3.event.x))));
    //     tf.gradient_stops.sort(function(a, b) { return a.t - b.t; });
    //     self.render();
    // });
    // path_gradients.exit().remove();

    var path_gaussians = this.svg.select("g.gaussians").selectAll("g").data(tf.gaussians);
    var path_gaussians_enter_g = path_gaussians.enter().append("g");
    path_gaussians_enter_g.append("path").attr("class", "sigma");
    path_gaussians_enter_g.append("circle").attr("class", "sigma_pos");
    path_gaussians_enter_g.append("circle").attr("class", "sigma_neg");
    path_gaussians_enter_g.append("path").attr("class", "symbol");
    path_gaussians_enter_g.attr("transform", "translate(0, " + (axis_y + 6) + ")");
    var drag_gaussians = d3.behavior.drag();
    var add_transform = function(selection) {
        selection.attr("transform", function(data) {
            var pos = scale(tf.inverse(data.center));
            return "translate(" + pos + ", 0)";
        });
    }
    path_gaussians.select("path.symbol")
      .attr("fill", function(data) { return rgba_color(data.color, 1); })
      .attr("d", glyph_droplet(12, 14))
      .call(add_transform)
      .call(drag_gaussians);
    path_gaussians.select("path.sigma")
      .attr("stroke", function(data) { return rgba_color(data.color, 1); })
      .attr("d", function(data) {
        return "M -% -32 L % -32 M 0 0 L 0 -32"
          .replace(/\%/g, (scale(tf.inverse(data.sigma + data.center)) - scale(tf.inverse(data.center))).toFixed(2));
      })
      .call(add_transform);
    var drag_gaussians_circle = d3.behavior.drag();
    path_gaussians.select("circle.sigma_pos")
      .attr("fill", function(data) { return rgba_color(data.color, 1); })
      .attr("cx", function(data) { return scale(tf.inverse(data.sigma + data.center)) - scale(tf.inverse(data.center)); })
      .attr("cy", -32)
      .attr("r", 5)
      .call(add_transform)
      .call(drag_gaussians_circle);
    path_gaussians.select("circle.sigma_neg")
      .attr("fill", function(data) { return rgba_color(data.color, 1); })
      .attr("cx", function(data) { return -(scale(tf.inverse(data.sigma + data.center)) - scale(tf.inverse(data.center))); })
      .attr("cy", -32)
      .attr("r", 5)
      .call(add_transform)
      .call(drag_gaussians_circle);
    drag_gaussians_circle.on("drag", function(data) {
        data.sigma = Math.abs(tf.scale(scale.invert(d3.event.x)) - data.center);
        self.render();
    });
    var drag_gaussians_context = null;
    drag_gaussians.on("dragstart", function(data) {
        drag_gaussians_context = {
            index: tf.gaussians.indexOf(data)
        };
    });
    drag_gaussians.on("drag", function(data) {
        data.center = tf.scale(scale.invert(d3.event.x));
        var dy = d3.event.y - 10;
        var index = tf.gaussians.indexOf(data);
        if(Math.abs(dy) > 100) {
            if(index >= 0) {
                tf.gaussians.splice(drag_gaussians_context.index, 1);
            }
        } else {
            if(index < 0) {
                tf.gaussians.splice(drag_gaussians_context.index, 0, data);
            }
        }
        self.render();
    });
    path_gaussians.exit().remove();

    var drag_gradient_max = d3.behavior.drag();
    this.svg.select("path.gradient_max").attr({
        "transform": "translate(" + (width + margin) + ", " + alpha_scale(tf.gradient_alpha_max) + ") rotate(-90)",
        "d": glyph_droplet(8, 8),
        "fill": "white"
    }).call(drag_gradient_max);
    drag_gradient_max.on("drag", function() {
        var y = d3.event.y;
        tf.gradient_alpha_max = Math.min(1, Math.max(0, alpha_scale.invert(y)));
        self.render();
    });

    var drag_gradient_power = d3.behavior.drag();
    if(this.drag_gradient_power_x === undefined)
        this.drag_gradient_power_x = 0.5;

    this.svg.select("path.gradient_power").attr({
        "transform": "translate(" + scale(tf.inverse(this.drag_gradient_power_x)) + ", " + alpha_scale(Math.pow(this.drag_gradient_power_x, tf.gradient_alpha_power) * tf.gradient_alpha_max) + ")",
        "d": glyph_droplet(-8, -8),
        "fill": "white"
    }).call(drag_gradient_power);
    drag_gradient_power.on("drag", function() {
        var y = d3.event.y;
        self.drag_gradient_power_x = tf.scale(scale.invert(d3.event.x));
        if(self.drag_gradient_power_x != self.drag_gradient_power_x) self.drag_gradient_power_x = 0.5;
        var p = Math.min(1, Math.max(0, alpha_scale.invert(y) / tf.gradient_alpha_max));
        var power = Math.log(p) / Math.log(self.drag_gradient_power_x);
        if(p == 0) power = 1;
        tf.gradient_alpha_power = power;
        self.render();
    });

    var ctx = this.canvas.node().getContext("2d");

    ctx.clearRect(0, 0, this.canvas.node().width, this.canvas.node().height);
    ctx.save();
    ctx.scale(2, 2);
    ctx.translate(margin, 0);
    ctx.lineJoin = "round";

    // Draw ticks.
    ctx.beginPath();
    alpha_scale.ticks(5).forEach(function(tick) {
        ctx.moveTo(0, alpha_scale(tick));
        ctx.lineTo(width, alpha_scale(tick));
    });

    scale.ticks(5).forEach(function(tick) {
        ctx.moveTo(scale(tick) - margin, alpha_scale(0));
        ctx.lineTo(scale(tick) - margin, alpha_scale(1));
    });
    ctx.strokeStyle = "rgba(255, 255, 255, 0.2)";
    ctx.lineWidth = 1;
    ctx.stroke();

    var gradient = ctx.createLinearGradient(0, 0, width, 0);
    tf.gradient_stops.forEach(function(stop) {
        gradient.addColorStop(stop.t, rgba_color(stop, 1));
    });
    ctx.fillStyle = gradient;
    ctx.fillRect(0, axis_y, width, 6);

    var gradient = ctx.createLinearGradient(0, 0, width, 0);
    ctx.beginPath();
    ctx.moveTo(0, alpha_scale(0));
    var N = 800;
    for(var i = 0; i < N; i++) {
        var t = i / (N - 1);
        var c = tf.sample(t);
        ctx.lineTo(t * width, alpha_scale(c.a));
        try {
            gradient.addColorStop(t, rgba_color(c));
        } catch(e) { }
    }
    ctx.lineTo(width, alpha_scale(0));
    ctx.fillStyle = "black";
    ctx.fill();
    ctx.fillStyle = gradient;
    ctx.fill();
    ctx.strokeStyle = "white";
    ctx.stroke();
    //ctx.fillRect(0, 0, width, axis_y);

    ctx.restore();

    var content = [];
    var N = 1600;
    for(var i = 0; i < N; i++) {
        var t = i / (N - 1);
        var c = tf.sample(t);
        content.push([ c.r, c.g, c.b, c.a ]);
    }
    if(this.onTransferFunctionChanged) {
        this.onTransferFunctionChanged({
            scale: tf.is_log ? "log" : "linear",
            domain: tf.domain,
            content: content
        });
    }
};
