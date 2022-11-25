using CUDA
using BenchmarkTools
using Plots

function odefcn(a,b,c,d,e,x,y)
    return   x^2 +   x*y +                 y,
            a*x^2 + b*x*y + c*y^2 + d*x + e*y
end

function RK4_limits(a, b, c, d, e, x0, y0, h_min, iter_max)
    
    xm2 = x0;
    ym2 = y0;
    xm1 = x0;
    ym1 = y0;
    
    k1x, k1y = odefcn(a, b, c, d, e, xm1, ym1);

    h = h_min;

    k2x, k2y = odefcn(a, b, c, d, e, xm1 + k1x * h/2, ym1 + k1y * h/2);
    k3x, k3y = odefcn(a, b, c, d, e, xm1 + k2x * h/2, ym1 + k2y * h/2);
    k4x, k4y = odefcn(a, b, c, d, e, xm1 + k3x * h  , ym1 + k3y * h  );
    
    x = xm1 + (h/6) * (k1x + 2*k2x + 2*k3x + k4x);
    y = ym1 + (h/6) * (k1y + 2*k2y + 2*k3y + k4y);
    
    count = 0;
    iter = 1;
    
    while count < 2 && iter < iter_max
        
        xm2 = xm1;
        ym2 = ym1;
        xm1 = x;
        ym1 = y;
        
        k1x, k1y = odefcn(a, b, c, d, e, xm1, ym1);
    
        h = h_min
    
        k2x, k2y = odefcn(a, b, c, d, e, xm1 + k1x * h/2, ym1 + k1y * h/2);
        k3x, k3y = odefcn(a, b, c, d, e, xm1 + k2x * h/2, ym1 + k2y * h/2);
        k4x, k4y = odefcn(a, b, c, d, e, xm1 + k3x * h  , ym1 + k3y * h  );

        x = xm1 + (h/6) * (k1x + 2*k2x + 2*k3x + k4x);
        y = ym1 + (h/6) * (k1y + 2*k2y + 2*k3y + k4y);
        
        if y*ym1 < 0
            count = count + 1;
        end

        iter = iter + 1;
        
    end
    
    xdiff = x0 - (xm2 * (ym1 * y  )   / ((ym2 - ym1) * (ym2 - y  )) +
                + xm1 * (ym2 * y  )   / ((ym1 - ym2) * (ym1 - y  )) +
                + x   * (ym2 * ym1)   / ((y   - ym2) * (y   - ym1)));


    
    return xdiff, iter
    
end

function gpu_RK4_limits!(a, b, c, d, e, points, xdiff, h, iter, iter_max)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    for i = index:stride:length(points)
        @inbounds xdiff[i], iter[i] = RK4_limits(a, b, c, d, e, points[i], 0.0, h, iter_max);
    end
    return nothing
end

function kernel_RK4_limits!(a, b, c, d, e, points, xdiff, h, iter, iter_max)
    kernel = @cuda launch=false gpu_RK4_limits!(a, b, c, d, e, points, xdiff, h, iter, iter_max)
    config = launch_configuration(kernel.fun)
    threads = min(length(points), config.threads)
    #threads = min(length(points), 1024)
    blocks = ceil(Int, length(points)/1024)


    CUDA.@sync begin
        @cuda threads=threads blocks=blocks gpu_RK4_limits!(a, b, c, d, e, points, xdiff, h, iter, iter_max)
    end
end

function compute_arrays!(a, b, c, d, e, RK_precision, search_start, search_end, search_step, iter_max)
    
    points = CuArray{Float32}(filter(!iszero, Array(search_start:search_step:search_end)));
    xdiff = CuArray{Float32}(CUDA.zeros(length(points)));
    iter = CuArray{Int}(CUDA.zeros(length(points)));
    kernel_RK4_limits!(a, b, c, d, e, points, xdiff, RK_precision, iter, iter_max)

    return Array(points), Array(xdiff), Array(iter);

end

function compute_arrays_atan!(a, b, c, d, e, RK_precision, atan_step, iter_max)

    #z = Array(-pi/2 + atan_step : atan_step : pi/2 - atan_step);
    z = Array(-pi/2  : atan_step : pi/2 );

    if z[lastindex(z)] > pi/2
        z = z[1:lastindex(z)-1];
    end

    #points = broadcast(atan, z);
    points = tan.(z)

    points = CuArray{Float32}(filter(!iszero, points));

    xdiff = CuArray{Float32}(CUDA.zeros(length(points)));
    iter = CuArray{Int}(CUDA.zeros(length(points)));
    kernel_RK4_limits!(a, b, c, d, e, points, xdiff, RK_precision, iter, iter_max)

    return Array(points), Array(xdiff), Array(iter);

end

function find_limit_cycles(points, xdiff, iter)

    drop_indices = findall(x->x>1e5,abs.(xdiff))
    deleteat!(xdiff, drop_indices);
    deleteat!(points, drop_indices);
    deleteat!(iter, drop_indices);

    cut_off = 1;

    negatives = xdiff[1:length(xdiff)-1].*xdiff[2:length(xdiff)].<=0;
    indices = findall(x->x==1, negatives);

    drop_indices = findall(x->x>cut_off, abs.(xdiff[indices]))
    deleteat!(indices, drop_indices);
    values = points[indices];
    iterations = iter[indices];

    return values, iterations, indices

end

a=-10;
b=2.2;
c=0.7;
d=-72-7/9;
e=0.0015;

h_d = 0.000001;
iter_max_d = 1000000;
start_d = -2;
end_d = 20;
step_d = 0.0001;

points_x, xdiff_x, iter_x = compute_arrays!(a, b, c, d, e, h_d, start_d, end_d, step_d, iter_max_d)

values, iterations, indices = find_limit_cycles(points_x, xdiff_x, iter_x)

RK4_limits(a, b, c, d, e, values[8], 0, h_d, iter_max_d)
RK4_limits(a, b, c, d, e, values[9], 0, h_d, iter_max_d)
RK4_limits(a, b, c, d, e, values[10], 0, h_d, iter_max_d)

points_x2, xdiff_x2, iter_x2 = compute_arrays!(a, b, c, d, e, h_d, 15, 16, 0.01, iter_max_d)

plot(points_x2, xdiff_x2)


points_t, xdiff_t, iter_t = compute_arrays_atan!(a, b, c, d, e, h_d, 0.0001, iter_max_d)

values, iterations, indices = find_limit_cycles(points_t, xdiff_t, iter_t)


values
values[5]

i = 5;
vecx, vecy = RK4_compute(a, b, c, d, e, values[i], 0.0, h_d, iterations[i]);
plt = plot(vecx, vecy)

vecx, vecy = RK4_compute(a, b, c, d, e, 0.001, 0.0, h_d, iterations[5]);
plt = plot(vecx, vecy)
vecx[1]-vecx[lastindex(vecx)]

function RK4_compute(a, b, c, d, e, x0, y0, h, iters)

    x = Array{Float64,1}(undef, iters+3)
    y = Array{Float64,1}(undef, iters+3)
    
    x[1] = x0;
    y[1] = y0;
    
    k1x, k1y = odefcn(a, b, c, d, e, x[1], y[1]);
    k2x, k2y = odefcn(a, b, c, d, e, x[1] + k1x * h/2, y[1] + k1y * h/2);
    k3x, k3y = odefcn(a, b, c, d, e, x[1] + k2x * h/2, y[1] + k2y * h/2);
    k4x, k4y = odefcn(a, b, c, d, e, x[1] + k3x * h  , y[1] + k3y * h  );
    
    x[2] = x[1] + (h/6) * (k1x + 2*k2x + 2*k3x + k4x);
    y[2] = y[1] + (h/6) * (k1y + 2*k2y + 2*k3y + k4y);
    
    for i in 3:iters+3
        
        k1x, k1y = odefcn(a, b, c, d, e, x[i-1], y[i-1]);
        k2x, k2y = odefcn(a, b, c, d, e, x[i-1] + k1x * h/2, y[i-1] + k1y * h/2);
        k3x, k3y = odefcn(a, b, c, d, e, x[i-1] + k2x * h/2, y[i-1] + k2y * h/2);
        k4x, k4y = odefcn(a, b, c, d, e, x[i-1] + k3x * h  , y[i-1] + k3y * h  );

        x[i] = x[i-1] + (h/6) * (k1x + 2*k2x + 2*k3x + k4x);
        y[i] = y[i-1] + (h/6) * (k1y + 2*k2y + 2*k3y + k4y);

    end
    x = x[[1:10:iters;]]
    y = y[[1:10:iters;]]
    
    return x, y
    
end

if length(values) > 0
    vecx, vecy = RK4_compute(a, b, c, d, e, values[1], 0.0, h_d, iterations[1]);
    plt = plot(vecx, vecy);
    for i in 2:lastindex(values)
        vecx, vecy = RK4_compute(a, b, c, d, e, values[i], 0.0, h_d, iterations[i]);
        plot!(plt,vecx, vecy);
    end
    plt
end

function calculate_area(x,y)
    heights = (abs.(y[1:lastindex(y)-1]) .+ abs.(y[2:lastindex(y)]))./2;
    widths = x[2:lastindex(y)] .- x[1:lastindex(y)-1];
    return sum(heights .* widths)
end

calculate_area([0,10,20], [10,10,10])

broadcast(atan,Array(1:1:10))
atan(pi/2)
tan(pi/2)


function RK4_crossing(a, b, c, d, e, x0, y0, h)
    
    xm2 = x0;
    ym2 = y0;
    xm1 = x0;
    ym1 = y0;
    
    k1x, k1y = odefcn(a, b, c, d, e, xm1, ym1);
    k2x, k2y = odefcn(a, b, c, d, e, xm1 + k1x * h/2, ym1 + k1y * h/2);
    k3x, k3y = odefcn(a, b, c, d, e, xm1 + k2x * h/2, ym1 + k2y * h/2);
    k4x, k4y = odefcn(a, b, c, d, e, xm1 + k3x * h  , ym1 + k3y * h  );
    
    x = xm1 + (h/6) * (k1x + 2*k2x + 2*k3x + k4x);
    y = ym1 + (h/6) * (k1y + 2*k2y + 2*k3y + k4y);
    
    count = 0;
    
    while count < 1
        
        xm2 = xm1;
        ym2 = ym1;
        xm1 = x;
        ym1 = y;
        
        k1x, k1y = odefcn(a, b, c, d, e, xm1, ym1);
        k2x, k2y = odefcn(a, b, c, d, e, xm1 + k1x * h/2, ym1 + k1y * h/2);
        k3x, k3y = odefcn(a, b, c, d, e, xm1 + k2x * h/2, ym1 + k2y * h/2);
        k4x, k4y = odefcn(a, b, c, d, e, xm1 + k3x * h  , ym1 + k3y * h  );

        x = xm1 + (h/6) * (k1x + 2*k2x + 2*k3x + k4x);
        y = ym1 + (h/6) * (k1y + 2*k2y + 2*k3y + k4y);
        
        if y*ym1 < 0
            count = count + 1;
        end
        
    end
    
    return x
    
end

function pair_up(a, b, c, d, e, values, y0, h)

    hoops = Array{Array{Float64},1}(undef, lastindex(values))

    for i in 1:lastindex(values)
        crossing  = RK4_crossing(a, b, c, d, e, values[i], y0, h)
        hoops[i] = [values[i],crossing]
    end
    
    return hoops
end

values[1]
asdf = RK4_crossing(a, b, c, d, e, values[8], 0, h_d)
asdf2 = RK4_crossing(a, b, c, d, e, asdf, 0, h_d)
abs(values[8]- asdf2)



hoops = pair_up(a, b, c, d, e, values, 0, h_d)

for i in 1:lastindex(values)
    print(values[i],"\t",hoops[i],"\n")
end