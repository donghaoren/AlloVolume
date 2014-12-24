var Slider = function(container, scale, min, max, value) {
    var self = this;

    this.value = value;

    var width = 800;
    var svg = container.append("svg").attr("class", "slider");
    svg.style("width", width + "px");
    svg.style("height", "35px");
    var scale = scale == "linear" ? d3.scale.linear() : d3.scale.log();
    scale.domain([ min, max ]);
    scale.range([ 20, width - 20 ]);
    var axis = d3.svg.axis().scale(scale).orient("bottom").tickPadding(3).innerTickSize(2).outerTickSize(4);
    svg.append("g").attr("transform", "translate(0, 23)").attr("class", "axis").call(axis);
    var control = svg.append("path").attr("d", glyph_droplet(-8, -12)).attr("fill", "#AAA");
    var render = function() {
        control.attr("transform", "translate(" + scale(self.value) + ", 23)");
    };
    var did_change = function() {
        if(self.onChange) {
            self.onChange(self.value);
        }
    };
    var drag = d3.behavior.drag();

    control.call(drag);
    drag.on("drag", function() {
        var x = d3.event.x;
        self.value = Math.max(min, Math.min(max, scale.invert(x)));
        did_change();
        render();
    });
    render();

    self.set = function(value) {
        self.value = value;
        render();
    };
};

var SliderLevels = function(container) {
    var self = this;

    this.min = 0;
    this.max = 1;
    this.pow = 1;

    var width = 800;
    var svg = container.append("svg").attr("class", "slider");
    svg.style("width", width + "px");
    svg.style("height", "35px");
    var scale = d3.scale.linear();
    scale.domain([ 0, 1 ]);
    scale.range([ 20, width - 20 ]);
    var axis = d3.svg.axis().scale(scale).orient("bottom").tickPadding(3).innerTickSize(2).outerTickSize(4);
    svg.append("g").attr("transform", "translate(0, 23)").attr("class", "axis").call(axis);
    var control_min = svg.append("path").attr("d", glyph_droplet(-8, -12)).attr("fill", "#AAA");
    var control_max = svg.append("path").attr("d", glyph_droplet(-8, -12)).attr("fill", "#AAA");
    var control_pow = svg.append("path").attr("d", glyph_droplet(-8, -12)).attr("fill", "#AAA");
    var render = function() {
        control_min.attr("transform", "translate(" + scale(self.min) + ", 23)");
        control_max.attr("transform", "translate(" + scale(self.max) + ", 23)");
        control_pow.attr("transform", "translate(" + scale(Math.pow(2, -1.0 / self.pow) * (self.max - self.min) + self.min) + ", 23)");
    };
    var did_change = function() {
        if(self.onChange) {
            self.onChange(self.min, self.max, self.pow);
        }
    };
    control_min.call(d3.behavior.drag().on("drag", function() {
        var x = d3.event.x;
        self.min = Math.max(0, Math.min(1, scale.invert(x)));
        did_change();
        render();
    }));
    control_max.call(d3.behavior.drag().on("drag", function() {
        var x = d3.event.x;
        self.max = Math.max(0, Math.min(1, scale.invert(x)));
        did_change();
        render();
    }));
    control_pow.call(d3.behavior.drag().on("drag", function() {
        var x = d3.event.x;
        self.pow = -Math.log(2) / Math.log((scale.invert(x) - self.min) / (self.max - self.min));
        if(self.pow != self.pow) self.pow = 10;
        if(self.pow > 100) self.pow = 100;
        did_change();
        render();
    }));
    self.set = function(min, max, pow) {
        self.min = min;
        self.max = max;
        self.pow = pow;
        render();
    };
    render();
};
