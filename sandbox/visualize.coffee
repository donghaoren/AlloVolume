data = require("./data.json")

graphics = require("allofw").graphics;
s = new graphics.Surface2D(1000, 1000, graphics.SURFACETYPE_RASTER);
context = new graphics.GraphicalContext2D(s);

paint = context.paint();
paint.setMode(graphics.PAINTMODE_STROKE)

context.clear(255, 255, 255, 1);

for x in [0..20]
    context.drawLine(x * 50, 0, x * 50, 1000, paint);
    context.drawLine(0, x * 50, 1000, x * 50, paint);

traversals = data.slice(0, data.length - 1)
p = data[data.length - 1].slice(0, 3)
d = data[data.length - 1].slice(3, 6)

context.drawLine(p[0] * 50, p[1] * 50, (p[0] + d[0] * 100) * 50, (p[1] + d[1] * 100) * 50, paint)

for [X, Y, Z, t1, t2] in traversals
    px = p[0] + d[0] * t1
    py = p[1] + d[1] * t1
    pz = p[2] + d[2] * t1
    px1 = p[0] + d[0] * t2
    py1 = p[1] + d[1] * t2
    pz1 = p[2] + d[2] * t2

    paint.setMode(graphics.PAINTMODE_FILL)
    context.drawCircle(px * 50, py * 50, 3, paint)
    context.drawCircle(px1 * 50, py1 * 50, 3, paint)
    paint.setMode(graphics.PAINTMODE_STROKE)
    paint.setColor(255, 0, 0, 1)
    context.drawRectangle(X * 50, Y * 50, 50, 50, paint)
    paint.setColor(0, 0, 0, 1)



s.save("visualize.png");
