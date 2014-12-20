// Transfer function editor.
var LayerTypes = { };

var TransferFunctionDescription = function() {
    this.domain = [ 1e-3, 1e8 ];
    this.scale = "log";
    this.layers = [];
    this.layers.push(LayerTypes["gaussians"].create(this));
};

LayerTypes["gaussians"] = {
    name: "Gaussians",
    type: "gaussians",
    // { t: "gaussians", t0: <float>, t1: <float>, gradient: [[r,g,b],...], ticks: <int>, alpha0: <float,0>, alpha1: <float,1>, sigma: <float,(t1-t0)/ticks/16> }
    create: function(tf) {
        var name;
        var names = tf.layers.map(function(x) { return x.n; });
        for(var i = 1;; i++) {
            name = "G" + i;
            if(names.indexOf(name) < 0) break;
        }
        return { n: name, t: "gaussians", gradient_name: "rainbow", gradient: Gradients["rainbow"].values, ticks: 10, t0: 0, t1: 1, alpha0: 0, alpha1: 0.9, sigma0: 1.0 / 16, sigma1: 1.0 / 16, alpha_pow: 1.0 };
    },
    sample: function(t, tf, layer) {
        var ticks = layer.ticks;
        var t0 = layer.t0;
        var t1 = layer.t1;
        var alpha0 = layer.alpha0; if(alpha0 === undefined) alpha0 = 0.1;
        var alpha1 = layer.alpha1; if(alpha1 === undefined) alpha1 = 0.9;
        var alpha_pow = layer.alpha_pow; if(alpha_pow === undefined) alpha_pow = 1.0;
        var sigma0 = layer.sigma0; if(sigma0 === undefined) sigma0 = 1.0 / 8.0;
        var sigma1 = layer.sigma1; if(sigma1 === undefined) sigma1 = 1.0 / 16.0;

        var color = { r: 0, g: 0, b: 0, a: 0 };

        for(var i = 0; i < ticks; i++) {
            var t_center = i / (ticks - 1) * (t1 - t0) + t0;
            var alpha = Math.pow(i / (ticks - 1), alpha_pow) * (alpha1 - alpha0) + alpha0;
            var sigma = i / (ticks - 1) * (sigma1 - sigma0) + sigma0;
            sigma = sigma * (t1 - t0) / ticks;
            var gaussian = Math.exp(- (t - t_center) / sigma * (t - t_center) / sigma / 2) * alpha;
            var c = sample_gradient(layer.gradient, i / (ticks - 1));
            c.a = gaussian;
            color = blend_color(c, color);
        }
        return color;
    },
    editor_layer: function(g, tf, layer, info) {
        var ticks = layer.ticks;
        var t0 = layer.t0;
        var t1 = layer.t1;
        var alpha0 = layer.alpha0; if(alpha0 === undefined) alpha0 = 0.1;
        var alpha1 = layer.alpha1; if(alpha1 === undefined) alpha1 = 0.9;
        var alpha_pow = layer.alpha_pow; if(alpha_pow === undefined) alpha_pow = 1.0;
        var sigma0 = layer.sigma0; if(sigma0 === undefined) sigma0 = 1.0 / 8.0;
        var sigma1 = layer.sigma1; if(sigma1 === undefined) sigma1 = 1.0 / 16.0;

        var path_t0 = g.selectAll("path.t0").data([0]);
        path_t0.enter().append("path").attr("class", "t0").attr("d", glyph_droplet(15, 15));
        var path_t1 = g.selectAll("path.t1").data([0]);
        path_t1.enter().append("path").attr("class", "t1").attr("d", glyph_droplet(15, 15));

        var text_name = g.selectAll("text.name").data([0]);
        text_name.enter().append("text").attr("class", "name").text(layer.n).attr("fill", "#AAA").attr("text-anchor", "middle");
        text_name.attr("transform", "translate(" + info.tscale(t0) + "," + (info.axis_y + 40) + ")")

        var path_sigma0 = g.selectAll("path.sigma0").data([0]);
        path_sigma0.enter().append("path").attr("class", "sigma0").attr("d", glyph_droplet(10, 10));
        var path_sigma1 = g.selectAll("path.sigma1").data([0]);
        path_sigma1.enter().append("path").attr("class", "sigma1").attr("d", glyph_droplet(10, 10));

        var path_alpha0 = g.selectAll("path.alpha0").data([0]);
        path_alpha0.enter().append("path").attr("class", "alpha0").attr("d", glyph_droplet(10, 10));
        var path_alpha1 = g.selectAll("path.alpha1").data([0]);
        path_alpha1.enter().append("path").attr("class", "alpha1").attr("d", glyph_droplet(10, 10));

        var path_alpha_pow = g.selectAll("path.alpha_pow").data([0]);
        path_alpha_pow.enter().append("path").attr("class", "alpha_pow").attr("d", glyph_droplet(10, 10));
        if(path_alpha_pow.node().x_pos === undefined) path_alpha_pow.node().x_pos = 0.5;

        path_t0.attr("transform", "translate(" + info.tscale(t0) + "," + (info.axis_y) + ")")
               .attr("fill", rgba_color(array_to_color(layer.gradient[0])));
        path_t1.attr("transform", "translate(" + info.tscale(t1) + "," + (info.axis_y) + ")")
               .attr("fill", rgba_color(array_to_color(layer.gradient[layer.gradient.length - 1])));

        path_sigma0.attr("transform", "translate(" + info.tscale(t0 + sigma0 * (t1 - t0) / layer.ticks * 3) + "," + (info.axis_y) + ")")
               .attr("fill", rgba_color(array_to_color(layer.gradient[0])));
        path_sigma1.attr("transform", "translate(" + info.tscale(t1 - sigma1 * (t1 - t0) / layer.ticks * 3) + "," + (info.axis_y) + ")")
               .attr("fill", rgba_color(array_to_color(layer.gradient[layer.gradient.length - 1])));

        path_alpha0.attr("transform", "translate(" + info.tscale(t0) + "," + info.alpha_scale(alpha0) + ") rotate(180)")
                   .attr("fill", rgba_color(array_to_color(layer.gradient[0])));
        path_alpha1.attr("transform", "translate(" + info.tscale(t1) + "," + info.alpha_scale(alpha1) + ") rotate(180)")
                   .attr("fill", rgba_color(array_to_color(layer.gradient[layer.gradient.length - 1])));

        var k = path_alpha_pow.node().x_pos;
        path_alpha_pow.attr("transform", "translate(" + info.tscale(t0 * (1 - k) + t1 * k) + "," + info.alpha_scale(Math.pow(path_alpha_pow.node().x_pos, alpha_pow) * (alpha1 - alpha0) + alpha0) + ") rotate(180)")
                   .attr("fill", rgba_color(sample_gradient(layer.gradient, k)));

        var drag_path_t0 = d3.behavior.drag();
        path_t0.call(drag_path_t0);
        drag_path_t0.on("drag", function() {
            var x = d3.event.x;
            layer.t0 = info.tscale.invert(x);
            info.did_edit_layer(layer);
        });

        var drag_path_t1 = d3.behavior.drag();
        path_t1.call(drag_path_t1);
        drag_path_t1.on("drag", function() {
            var x = d3.event.x;
            layer.t1 = info.tscale.invert(x);
            info.did_edit_layer(layer);
        });

        var drag_path_sigma0 = d3.behavior.drag();
        path_sigma0.call(drag_path_sigma0);
        drag_path_sigma0.on("drag", function() {
            var x = d3.event.x;
            layer.sigma0 = Math.abs((info.tscale.invert(x) - layer.t0) / 3 * layer.ticks / (t1 - t0));
            info.did_edit_layer(layer);
        });

        var drag_path_sigma1 = d3.behavior.drag();
        path_sigma1.call(drag_path_sigma1);
        drag_path_sigma1.on("drag", function() {
            var x = d3.event.x;
            layer.sigma1 = Math.abs((info.tscale.invert(x) - layer.t1) / 3 * layer.ticks / (t1 - t0));
            info.did_edit_layer(layer);
        });

        var drag_path_alpha0 = d3.behavior.drag();
        path_alpha0.call(drag_path_alpha0);
        drag_path_alpha0.on("drag", function() {
            var y = d3.event.y;
            layer.alpha0 = clamp(info.alpha_scale.invert(y), 0, 0.999);
            info.did_edit_layer(layer);
        });

        var drag_path_alpha1 = d3.behavior.drag();
        path_alpha1.call(drag_path_alpha1);
        drag_path_alpha1.on("drag", function() {
            var y = d3.event.y;
            layer.alpha1 = clamp(info.alpha_scale.invert(y), 0, 0.999);
            info.did_edit_layer(layer);
        });

        var drag_path_alpha_pow = d3.behavior.drag();
        path_alpha_pow.call(drag_path_alpha_pow);
        drag_path_alpha_pow.on("drag", function() {
            var x = d3.event.x;
            var y = d3.event.y;
            path_alpha_pow.node().x_pos = (info.tscale.invert(x) - t0) / (t1 - t0);
            var new_val = Math.log((info.alpha_scale.invert(y) - alpha0) / (alpha1 - alpha0)) / Math.log(path_alpha_pow.node().x_pos);
            if(new_val != new_val) new_val = 100;
            if(new_val > 100) new_val = 100;
            if(new_val < 0.01) new_val = 0.01;
            layer.alpha_pow = new_val;
            info.did_edit_layer(layer);
        });
    },
    editor_interface: function(div, tf, layer, info) {
        var text_ticks = div.selectAll("input.ticks").data([0]);
        text_ticks.enter().append("input").attr("class", "ticks").attr("type", "text");
        text_ticks.node().value = layer.ticks.toString();
        text_ticks.on("change", function() {
            layer.ticks = parseInt(text_ticks.node().value);
            if(layer.ticks < 2) layer.ticks = 2;
            info.did_edit_layer(layer);
        });

        var btn_add_tick = div.selectAll("span.btn.addtick").data([0]);
        btn_add_tick.enter().append("span").attr("class", "btn addtick").text("+");
        btn_add_tick.on("click", function() {
            layer.ticks += 1;
            info.did_edit_layer(layer);
        });

        var btn_remove_tick = div.selectAll("span.btn.removetick").data([0]);
        btn_remove_tick.enter().append("span").attr("class", "btn removetick").text("−");
        btn_remove_tick.on("click", function() {
            layer.ticks -= 1;
            if(layer.ticks < 2) layer.ticks = 2;
            info.did_edit_layer(layer);
        });

        var gradient_select = div.selectAll("select.gradient").data([0]);
        gradient_select.enter().append("select").attr("class", "gradient");
        gradient_select.each(function() {
            var item = d3.select(this);
            var gradients = [];
            for(var key in Gradients) {
                var g = Gradients[key];
                gradients.push({
                    name: g.name,
                    key: key
                });
            }
            item.selectAll("option").data(gradients).enter()
              .append("option")
                .attr("value", function(d) { return d.key; })
                .text(function(d) { return d.name; });
            item.node().value = layer.gradient_name;
            item.on("change", function() {
                var v = item.node().value;
                layer.gradient = Gradients[v].values;
                layer.gradient_name = v;
                info.did_edit_layer(layer);
            });
        });
    }
};

LayerTypes["gradient"] = {
    name: "Gradient",
    type: "gradient",
    create: function(tf) {
        var name;
        var names = tf.layers.map(function(x) { return x.n; });
        for(var i = 1;; i++) {
            name = "G" + i;
            if(names.indexOf(name) < 0) break;
        }
        return { n: name, t: "gradient", gradient_name: "rainbow", gradient: Gradients["rainbow"].values, t0: 0, t1: 1, alpha0: 0.1, alpha1: 0.9, alpha_pow: 1.0 };
    },
    sample: function(t, tf, layer) {
        var t0 = layer.t0;
        var t1 = layer.t1;
        var alpha0 = layer.alpha0; if(alpha0 === undefined) alpha0 = 0.1;
        var alpha1 = layer.alpha1; if(alpha1 === undefined) alpha1 = 0.9;
        var alpha_pow = layer.alpha_pow; if(alpha_pow === undefined) alpha_pow = 1.0;
        if(t < t0 || t > t1) return { r: 0, g: 0, b: 0, a: 0 };
        var c = sample_gradient(layer.gradient, (t - t0) / (t1 - t0));
        c.a = Math.pow((t - t0) / (t1 - t0), alpha_pow) * (alpha1 - alpha0) + alpha0;
        return c;
    },
    editor_layer: function(g, tf, layer, info) {
        var ticks = layer.ticks;
        var t0 = layer.t0;
        var t1 = layer.t1;
        var alpha0 = layer.alpha0; if(alpha0 === undefined) alpha0 = 0.1;
        var alpha1 = layer.alpha1; if(alpha1 === undefined) alpha1 = 0.9;
        var alpha_pow = layer.alpha_pow; if(alpha_pow === undefined) alpha_pow = 1.0;

        var path_t0 = g.selectAll("path.t0").data([0]);
        path_t0.enter().append("path").attr("class", "t0").attr("d", glyph_droplet(15, 15));
        var path_t1 = g.selectAll("path.t1").data([0]);
        path_t1.enter().append("path").attr("class", "t1").attr("d", glyph_droplet(15, 15));

        var text_name = g.selectAll("text.name").data([0]);
        text_name.enter().append("text").attr("class", "name").text(layer.n).attr("fill", "#AAA").attr("text-anchor", "middle");
        text_name.attr("transform", "translate(" + info.tscale(t0) + "," + (info.axis_y + 40) + ")")

        var path_alpha0 = g.selectAll("path.alpha0").data([0]);
        path_alpha0.enter().append("path").attr("class", "alpha0").attr("d", glyph_droplet(10, 10));
        var path_alpha1 = g.selectAll("path.alpha1").data([0]);
        path_alpha1.enter().append("path").attr("class", "alpha1").attr("d", glyph_droplet(10, 10));

        var path_alpha_pow = g.selectAll("path.alpha_pow").data([0]);
        path_alpha_pow.enter().append("path").attr("class", "alpha_pow").attr("d", glyph_droplet(10, 10));
        if(path_alpha_pow.node().x_pos === undefined) path_alpha_pow.node().x_pos = 0.5;

        path_t0.attr("transform", "translate(" + info.tscale(t0) + "," + (info.axis_y) + ")")
               .attr("fill", rgba_color(array_to_color(layer.gradient[0])));
        path_t1.attr("transform", "translate(" + info.tscale(t1) + "," + (info.axis_y) + ")")
               .attr("fill", rgba_color(array_to_color(layer.gradient[layer.gradient.length - 1])));

        path_alpha0.attr("transform", "translate(" + info.tscale(t0) + "," + info.alpha_scale(alpha0) + ") rotate(180)")
                   .attr("fill", rgba_color(array_to_color(layer.gradient[0])));
        path_alpha1.attr("transform", "translate(" + info.tscale(t1) + "," + info.alpha_scale(alpha1) + ") rotate(180)")
                   .attr("fill", rgba_color(array_to_color(layer.gradient[layer.gradient.length - 1])));

        var k = path_alpha_pow.node().x_pos;
        path_alpha_pow.attr("transform", "translate(" + info.tscale(t0 * (1 - k) + t1 * k) + "," + info.alpha_scale(Math.pow(path_alpha_pow.node().x_pos, alpha_pow) * (alpha1 - alpha0) + alpha0) + ") rotate(180)")
                   .attr("fill", rgba_color(sample_gradient(layer.gradient, k)));

        var drag_path_t0 = d3.behavior.drag();
        path_t0.call(drag_path_t0);
        drag_path_t0.on("drag", function() {
            var x = d3.event.x;
            layer.t0 = info.tscale.invert(x);
            info.did_edit_layer(layer);
        });

        var drag_path_t1 = d3.behavior.drag();
        path_t1.call(drag_path_t1);
        drag_path_t1.on("drag", function() {
            var x = d3.event.x;
            layer.t1 = info.tscale.invert(x);
            info.did_edit_layer(layer);
        });

        var drag_path_alpha0 = d3.behavior.drag();
        path_alpha0.call(drag_path_alpha0);
        drag_path_alpha0.on("drag", function() {
            var y = d3.event.y;
            layer.alpha0 = clamp(info.alpha_scale.invert(y), 0, 0.999);
            info.did_edit_layer(layer);
        });

        var drag_path_alpha1 = d3.behavior.drag();
        path_alpha1.call(drag_path_alpha1);
        drag_path_alpha1.on("drag", function() {
            var y = d3.event.y;
            layer.alpha1 = clamp(info.alpha_scale.invert(y), 0, 0.999);
            info.did_edit_layer(layer);
        });

        var drag_path_alpha_pow = d3.behavior.drag();
        path_alpha_pow.call(drag_path_alpha_pow);
        drag_path_alpha_pow.on("drag", function() {
            var x = d3.event.x;
            var y = d3.event.y;
            path_alpha_pow.node().x_pos = (info.tscale.invert(x) - t0) / (t1 - t0);
            var new_val = Math.log((info.alpha_scale.invert(y) - alpha0) / (alpha1 - alpha0)) / Math.log(path_alpha_pow.node().x_pos);
            if(new_val != new_val) new_val = 100;
            if(new_val > 100) new_val = 100;
            if(new_val < 0.01) new_val = 0.01;
            layer.alpha_pow = new_val;
            info.did_edit_layer(layer);
        });
    },
    editor_interface: function(div, tf, layer, info) {
        var gradient_select = div.selectAll("select.gradient").data([0]);
        gradient_select.enter().append("select").attr("class", "gradient");
        gradient_select.each(function() {
            var item = d3.select(this);
            var gradients = [];
            for(var key in Gradients) {
                var g = Gradients[key];
                gradients.push({
                    name: g.name,
                    key: key
                });
            }
            item.selectAll("option").data(gradients).enter()
              .append("option")
                .attr("value", function(d) { return d.key; })
                .text(function(d) { return d.name; });
            item.node().value = layer.gradient_name;
            item.on("change", function() {
                var v = item.node().value;
                layer.gradient = Gradients[v].values;
                layer.gradient_name = v;
                info.did_edit_layer(layer);
            });
        });
    }
};

LayerTypes["block"] = {
    name: "Block",
    type: "block",
    create: function(tf) {
        var name;
        var names = tf.layers.map(function(x) { return x.n; });
        for(var i = 1;; i++) {
            name = "B" + i;
            if(names.indexOf(name) < 0) break;
        }
        return { n: name, t: "block", color: Colors[4].concat([0.9]), tm: 0.5, span: 0.1, feather: 0.005 };
    },
    sample: function(t, tf, layer) {
        var r = array_to_color(layer.color);
        var sigmoid = function(t) { return 1.0 / (1.0 + Math.exp(-t / layer.feather)); };
        r.a *= (sigmoid(t - (layer.tm - layer.span)) + sigmoid(layer.tm + layer.span - t) - 1.0);
        return r;
    },
    editor_layer: function(g, tf, layer, info) {
        var path_tm = g.selectAll("path.tm").data([0]);
        path_tm.enter().append("path").attr("class", "tm").attr("d", glyph_droplet(15, 15));

        var path_span1 = g.selectAll("path.span1").data([0]);
        path_span1.enter().append("path").attr("class", "span1").attr("d", glyph_droplet(15, 10));
        var path_span2 = g.selectAll("path.span2").data([0]);
        path_span2.enter().append("path").attr("class", "span2").attr("d", glyph_droplet(15, 10));

        var path_feather1 = g.selectAll("path.feather1").data([0]);
        path_feather1.enter().append("path").attr("class", "feather1").attr("d", glyph_droplet(10, 8));
        var path_feather2 = g.selectAll("path.feather2").data([0]);
        path_feather2.enter().append("path").attr("class", "feather2").attr("d", glyph_droplet(10, 8));

        var path_alpha = g.selectAll("path.alpha").data([0]);
        path_alpha.enter().append("path").attr("class", "alpha").attr("d", glyph_droplet(10, 10));

        var text_name = g.selectAll("text.name").data([0]);
        text_name.enter().append("text").attr("class", "name").text(layer.n).attr("fill", "#AAA").attr("text-anchor", "middle");
        text_name.attr("transform", "translate(" + info.tscale(layer.tm) + "," + (info.axis_y + 40) + ")")

        path_tm.attr("transform", "translate(" + info.tscale(layer.tm) + "," + (info.axis_y) + ")")
               .attr("fill", rgba_color(array_to_color(layer.color.slice(0, 3))));

        path_span1.attr("transform", "translate(" + info.tscale(layer.span + layer.tm) + "," + (info.axis_y) + ")")
               .attr("fill", rgba_color(array_to_color(layer.color.slice(0, 3))));

        path_span2.attr("transform", "translate(" + info.tscale(-layer.span + layer.tm) + "," + (info.axis_y) + ")")
               .attr("fill", rgba_color(array_to_color(layer.color.slice(0, 3))));

        path_feather1.attr("transform", "translate(" + info.tscale(layer.span + layer.feather * 5 + layer.tm) + "," + (info.axis_y) + ")")
               .attr("fill", rgba_color(array_to_color(layer.color.slice(0, 3))));

        path_feather2.attr("transform", "translate(" + info.tscale(-layer.span - layer.feather * 5 + layer.tm) + "," + (info.axis_y) + ")")
               .attr("fill", rgba_color(array_to_color(layer.color.slice(0, 3))));



        path_alpha.attr("transform", "translate(" + info.tscale(layer.tm) + "," + info.alpha_scale(layer.color[3]) + ") rotate(180)")
                   .attr("fill", rgba_color(array_to_color(layer.color.slice(0, 3))));

        var drag_path_tm = d3.behavior.drag();
        path_tm.call(drag_path_tm);
        drag_path_tm.on("drag", function() {
            var x = d3.event.x;
            layer.tm = info.tscale.invert(x);
            info.did_edit_layer(layer);
        });

        var drag_path_span1 = d3.behavior.drag();
        path_span1.call(drag_path_span1);
        drag_path_span1.on("drag", function() {
            var x = d3.event.x;
            layer.span = Math.abs(info.tscale.invert(x) - layer.tm);
            info.did_edit_layer(layer);
        });
        var drag_path_span2 = d3.behavior.drag();
        path_span2.call(drag_path_span2);
        drag_path_span2.on("drag", function() {
            var x = d3.event.x;
            layer.span = Math.abs(info.tscale.invert(x) - layer.tm);
            info.did_edit_layer(layer);
        });
        var drag_path_feather1 = d3.behavior.drag();
        path_feather1.call(drag_path_feather1);
        drag_path_feather1.on("drag", function() {
            var x = d3.event.x;
            layer.feather = Math.abs(info.tscale.invert(x) - layer.tm - layer.span) / 5;
            info.did_edit_layer(layer);
        });
        var drag_path_feather2 = d3.behavior.drag();
        path_feather2.call(drag_path_feather2);
        drag_path_feather2.on("drag", function() {
            var x = d3.event.x;
            layer.feather = Math.abs(layer.span + (info.tscale.invert(x) - layer.tm)) / 5;
            info.did_edit_layer(layer);
        });
        var drag_path_alpha = d3.behavior.drag();
        path_alpha.call(drag_path_alpha);
        drag_path_alpha.on("drag", function() {
            var y = d3.event.y;
            layer.color[3] = clamp(info.alpha_scale.invert(y), 0, 0.999);
            info.did_edit_layer(layer);
        });
    },
    on_select_color: function(color, tf, layer, info) {
        layer.color[0] = color[0];
        layer.color[1] = color[1];
        layer.color[2] = color[2];
        info.did_edit_layer(layer);
    },
    editor_interface: function() {
    }
};

TransferFunctionDescription.prototype.sample = function(t) {
    var color = { r: 0, g: 0, b: 0, a: 0 };
    this.layers.forEach(function(layer) {
        var c = LayerTypes[layer.t].sample(t, this, layer);
        color = blend_color(c, color);
    });
    return color;
};

TransferFunctionDescription.prototype.scale = function(value) {
    var td = this.domain;
    var tv = value;
    if(this.scale == "log") {
        td = [ Math.log(td[0]), Math.log(td[1]) ];
        tv = Math.log(tv);
    }
    return (tv - td[0]) / (td[1] - td[0]);
};

TransferFunctionDescription.prototype.inverse = function(value) {
    var td = this.domain;
    if(this.scale == "log") {
        td = [ Math.log(td[0]), Math.log(td[1]) ];
    }
    var tv = value * (td[1] - td[0]) + td[0];
    if(this.scale == "log") {
        tv = Math.exp(tv);
    }
    return tv;
};

// TransferFunctionDescription.prototype.sampleGradient = function(t) {
//     for(var i = 0; i < this.gradient_stops.length - 1; i++) {
//         var t0 = this.gradient_stops[i].t;
//         var t1 = this.gradient_stops[i + 1].t;
//         if(t0 == t1) continue;
//         if(t0 <= t && t <= t1) {
//             var p = (t - t0) / (t1 - t0);
//             var c1 = this.gradient_stops[i], c2 = this.gradient_stops[i + 1];
//             var alpha = this.gradient_alpha_max * Math.pow(t, this.gradient_alpha_power);
//             return {
//                 r: c1.r * (1 - p) + c2.r * p,
//                 g: c1.g * (1 - p) + c2.g * p,
//                 b: c1.b * (1 - p) + c2.b * p,
//                 a: (c1.a * (1 - p) + c2.a * p) * alpha,
//             };
//         }
//     }
//     return {
//         r: 0, g: 0, b: 0, a: 0
//     };
// };

// TransferFunctionDescription.prototype.sampleGaussian = function(t) {
//     var c = null;
//     for(var i = 0; i < this.gaussians.length; i++) {
//         var g = this.gaussians[i];
//         var a = g.color.a * Math.exp(-Math.pow((g.center - t) / g.sigma, 2) / 2);
//         var color = { r: g.color.r, g: g.color.g, b: g.color.b, a: a };
//         if(c) c = blend_color(color, c);
//         else c = color;
//     }
//     return c;
// };

// TransferFunctionDescription.prototype.sample = function(t) {
//     var g1 = this.sampleGradient(t);
//     var g2 = this.sampleGaussian(t);
//     if(g2) {
//         return blend_color(g2, g1);
//     } else {
//         return g1;
//     }
// };

// TransferFunctionDescription.prototype.generateTexture = function(size) {
//     if(!size) size = 1600;
//     var samples = [];
//     for(var i = 0; i < size; i++) {
//         var t = (i + 0.5) / size;
//         var r = this.sample(t);
//         samples.push([ r.r, r.g, r.b, r.a ]);
//     }
//     return samples;
// };

// TransferFunctionDescription.prototype.generateGradient = function(gradient, alpha_max, alpha_pow) {
//     if(alpha_max === undefined) alpha_max = 0.1;
//     if(alpha_pow === undefined) alpha_pow = 1;
//     this.gradient_alpha_max = alpha_max;
//     this.gradient_alpha_power = alpha_pow;
//     var self = this;
//     if(typeof(gradient) == "string") {
//         gradient = TransferFunctionDescription.Gradients[gradient];
//     }
//     if(gradient.type == "uniform") {
//         var length = gradient.values.length;
//         this.gradient_stops = [];
//         for(var index = 0; index < gradient.values.length; index++) {
//             var rgb = gradient.values[index];
//             var t = index / (length - 1);
//             this.gradient_stops.push({ t: t, r: rgb[0], g: rgb[1], b: rgb[2], a: rgb.length == 4 ? rgb[3] : 1 });
//         }
//     }
// };

// TransferFunctionDescription.prototype.generateGaussians = function(tmin, tmax, count) {
//     for(var i = 1; i <= count; i++) {
//         var t = i / (count) * (tmax - tmin) + tmin;
//         var c = this.sampleGradient(t);
//         c.a = this.sampleGradient(t).a;
//         this.gaussians.push({
//             center: t,
//             sigma: (tmax - tmin) / (count) / 15,
//             color: c
//         });
//     }
// };

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

    this.svg.append("g").attr("class", "axis");
    this.svg.append("g").attr("class", "axis_upper axis");
    this.svg.append("g").attr("class", "axis_alpha axis");
    this.g_layers = this.svg.append("g").attr("class", "layers");

    this.buttons = this.controls.append("div");
    this.color_picker = this.controls.append("div");
    this.layer_editors = this.controls.append("div");

    this.color_picker.selectAll("span.color").data(Colors).enter().append("span")
        .attr("class", "color btn")
        .style({
            "display": "inline-block",
            "width": "1em",
            "height": "1em",
            "margin": "0.4em 2px"
        })
        .style("background-color", function(d) { return rgba_color(array_to_color(d)) })
        .on("click", function(d) {
            if(self.on_select_color) self.on_select_color(d);
        });

    this.buttons.append("span").text("Min: ");
    this.range_min_input = this.buttons.append("input").attr("type", "text");
    this.buttons.append("span").text("Max: ");
    this.range_max_input = this.buttons.append("input").attr("type", "text");
    this.buttons.append("span")
      .attr("class", "btn")
      .text("Range")
      .on("click", function() {
        self.tf.domain = [ parseFloat(self.range_min_input.node().value), parseFloat(self.range_max_input.node().value) ];
        self.render();
      });

    this.buttons.append("span").text(" ").attr("class", "sep");
    this.buttons.append("span").text("Layer Type: ");
    this.select_layer_type = this.buttons.append("select");
    types = [];
    for(var key in LayerTypes) { types.push(LayerTypes[key]); }
    this.select_layer_type.selectAll("option").data(types).enter()
      .append("option")
        .attr("value", function(d) { return d.type; })
        .text(function(d) { return d.name; });
    this.buttons.append("span")
      .attr("class", "btn")
      .text("Add Layer")
      .on("click", function() {
        self.tf.layers.push(LayerTypes[self.select_layer_type.node().value].create(self.tf));
        self.did_edit();
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

    var axis_y = this.canvas_height - 40;
    var axis_y_upper = axis_y - 26;

    var margin = 40;
    var width = this.width - margin * 2;

    var scale = d3.scale.linear();
    if(tf.scale == "log") {
        scale = d3.scale.log().base(10);
    }

    scale
      .domain(tf.domain)
      .range([ margin, width + margin ]);

    var tscale = d3.scale.linear().domain([0, 1]).range([ margin, width + margin ]);

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

    var info = {
        axis_y: axis_y,
        alpha_scale: alpha_scale,
        scale: scale,
        tscale: tscale,
        did_edit_layer: function(layer) {
            self.selected_layer = layer;
            self.did_edit();
            self.render();
        }
    };

    var layer_id = function(l) {
        return l.type + ";" + l.n;
    };

    var layers = this.g_layers.selectAll("g.layerraphics").data(tf.layers, layer_id);
    layers.enter().append("g").attr("class", "layerraphics");
    layers.each(function(d, i) {
        LayerTypes[d.t].editor_layer(d3.select(this), tf, d, info);
    });
    layers.exit().remove();

    layers.on("click", function(d) {
        self.selected_layer = d;
        self.render();
    });


    var div_layer_editors = this.layer_editors.selectAll("div.layereditor").data(tf.layers, layer_id);
    div_layer_editors.enter().append("div").attr("class", "layereditor").style("margin", "10px 0px");
    div_layer_editors.each(function(d, i) {
        var item = d3.select(this);
        var span_type = item.selectAll("span.type").data([0]);
        span_type.enter().append("span").attr("class", "type").text(d.n + " (" + LayerTypes[d.t].name + "): ");

        span_type.style("color", d == self.selected_layer ? "white" : "inherit");

        LayerTypes[d.t].editor_interface(item, tf, d, info);
        var span_remove = item.selectAll("span.btn.remove").data([0]);
        span_remove.enter().append("span").attr("class", "remove btn").text("Remove");
        span_remove.on("click", function() {
            var idx = tf.layers.indexOf(d);
            if(idx >= 0) {
                tf.layers.splice(idx, 1);
                self.did_edit();
                self.render();
            }
        });
    });
    div_layer_editors.exit().remove();

    self.on_select_color = function(color) {
        if(self.selected_layer) {
            if(LayerTypes[self.selected_layer.t].on_select_color) {
                LayerTypes[self.selected_layer.t].on_select_color(color, tf, self.selected_layer, info);
            }
        }
    };

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

    ctx.restore();
};

TransferFunctionEditor.prototype.did_edit = function() {
    var tf = this.tf;
    if(this.onTransferFunctionChanged) {
        this.onTransferFunctionChanged({
            scale: tf.scale,
            domain: tf.domain,
            layers: tf.layers
        });
    }
};
