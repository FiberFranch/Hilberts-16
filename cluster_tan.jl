using CUDA
using BenchmarkTools
using Plots

function odefcn(a,b,c,d,e,x,y)
    return   x^2 +   x*y +                 y,
            a*x^2 + b*x*y + c*y^2 + d*x + e*y
end

function RK4_limits(a, b, c, d, e, x0, y0, h, iter_max)
    
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
    iter = 1;
    
    while count < 2 && iter < iter_max
        
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

function compute_arrays_tan!(a, b, c, d, e, RK_precision, atan_step, iter_max)

    #z = Array(-pi/2 + atan_step : atan_step : pi/2 - atan_step);
    z = Array(-pi/2 : atan_step : pi/2 );

    #=
    if z[lastindex(z)] > pi/2
        z = z[1:lastindex(z)-1];
    end
    =#

    z = z[2:lastindex(z)-1];

    #points = broadcast(atan, z);
    points = tan.(z)

    points = CuArray{Float32}(filter(!iszero, points));

    xdiff = CuArray{Float32}(CUDA.zeros(length(points)));
    iter = CuArray{Int}(CUDA.zeros(length(points)));
    kernel_RK4_limits!(a, b, c, d, e, points, xdiff, RK_precision, iter, iter_max)

    return Array(points), Array(xdiff), Array(iter);

end

function find_possible_cycles(points, xdiff, iter)

    drop_indices = findall(x->x>1e5,abs.(xdiff))
    deleteat!(xdiff, drop_indices);
    deleteat!(points, drop_indices);
    deleteat!(iter, drop_indices);

    negatives = xdiff[1:length(xdiff)-1].*xdiff[2:length(xdiff)].<=0;
    indices = findall(x->x==1, negatives);
    values = points[indices];
    iterations = iter[indices];

    return values, iterations, indices

end

function close_in(a, b, c, d, e, h, iter_max, x, y, xdiff, iter)
    xdiff_min = xdiff;
    x_min = x;
    iter_min = iter;

    x_prev = x;
    dx = 0.0025*abs(x);

    for i in 1:40

        x_act = x_prev + dx;
        xdiff_act, iter_act = RK4_limits(a, b, c, d, e, x_act, y, h, iter_max);

        if abs(xdiff_act) < abs(xdiff_min)

            xdiff_min = xdiff_act
            x_min = x_act;
            iter_min = iter_act;

        end

        x_prev = x_act;
    end

    return x_min, iter_min, xdiff_min

end

function find_limit_cycles(a, b, c, d, e, h, iter_max, points, xdiff, iter)

    values, iterations, indices = find_possible_cycles(points, xdiff, iter);
    p_diff = Array{Float64,1}(undef, lastindex(values));

    for i in 1:lastindex(values)
        values[i], iterations[i], p_diff[i] = close_in(a, b, c, d, e, h, iter_max, values[i], 0, xdiff[indices[i]], iterations[i]);
    end


    cut_off = 100;
    drop_indices = findall(x->x>cut_off, abs.(p_diff))
    deleteat!(values, drop_indices);
    deleteat!(iterations, drop_indices);
    deleteat!(p_diff, drop_indices);

    values_rep = round.(values, digits=3);
    #values_rep = round.(values, sigdigits=3);
    drop_indices = [];

    for i in 1:lastindex(values_rep)-1
        if values_rep[i]==values_rep[i+1]
            push!(drop_indices, i+1)
        end
    end


    drop_indices = drop_indices[unique(i -> drop_indices[i], 1:length(drop_indices))]
    
    deleteat!(values, drop_indices);
    deleteat!(iterations, drop_indices);
    deleteat!(p_diff, drop_indices);
    
    drop_indices = findall(x->x<=1e-5, abs.(values))
    deleteat!(values, drop_indices);
    deleteat!(iterations, drop_indices);
    deleteat!(p_diff, drop_indices);

    return values, iterations, p_diff

end

a=-10;
b=2.2;
c=0.7;
d=-72-7/9;
e=0.0015;

h_d = 0.000001;
iter_max_d = 1000000;
#start_d = -4000;
#end_d = -3000;
#step_d = 0.1;


points_x, xdiff_x, iter_x = compute_arrays!(a, b, c, d, e, h_d, 6900, 7100, 1, iter_max_d)

plot(points_x, xdiff_x, legend = false)


@btime points_t, xdiff_t, iter_t = compute_arrays_tan!(a, b, c, d, e, h_d, 0.1, iter_max_d);
points_t, xdiff_t, iter_t = compute_arrays_tan!(a, b, c, d, e, h_d, 0.1, iter_max_d);
length(points_t)
points_t[1]
points_t[lastindex(points_t)]

@btime points_t, xdiff_t, iter_t = compute_arrays_tan!(a, b, c, d, e, h_d, 0.01, iter_max_d);
points_t, xdiff_t, iter_t = compute_arrays_tan!(a, b, c, d, e, h_d, 0.01, iter_max_d);
length(points_t)
points_t[1]
points_t[lastindex(points_t)]

@btime points_t, xdiff_t, iter_t = compute_arrays_tan!(a, b, c, d, e, h_d, 0.001, iter_max_d);
points_t, xdiff_t, iter_t = compute_arrays_tan!(a, b, c, d, e, h_d, 0.001, iter_max_d);
length(points_t)
points_t[1]
points_t[lastindex(points_t)]

@btime points_t, xdiff_t, iter_t = compute_arrays_tan!(a, b, c, d, e, h_d, 0.0001, iter_max_d);
points_t, xdiff_t, iter_t = compute_arrays_tan!(a, b, c, d, e, h_d, 0.0001, iter_max_d);
length(points_t)
points_t[1]
points_t[lastindex(points_t)]

@btime points_t, xdiff_t, iter_t = compute_arrays_tan!(a, b, c, d, e, h_d, 0.00001, iter_max_d);
points_t, xdiff_t, iter_t = compute_arrays_tan!(a, b, c, d, e, h_d, 0.00001, iter_max_d);
length(points_t)
points_t[1]
points_t[lastindex(points_t)]