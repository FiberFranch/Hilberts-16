using CUDA
CUDA.versioninfo()

function odefcn(x,y)
    a=-10;
    b=2.2;
    c=0.7;
    d=-72-7/9;
    e=0.0015;
    
    return   x^2 +   x*y +                 y,
            a*x^2 + b*x*y + c*y^2 + d*x + e*y
end

function RK4(x0, y0, h, iter)
    
    xm2 = x0;
    ym2 = y0;
    xm1 = x0;
    ym1 = y0;
    
    k1x, k1y = odefcn(xm1, ym1);
    k2x, k2y = odefcn(xm1 + k1x * h/2, ym1 + k1y * h/2);
    k3x, k3y = odefcn(xm1 + k2x * h/2, ym1 + k2y * h/2);
    k4x, k4y = odefcn(xm1 + k3x * h  , ym1 + k3y * h  );
    
    x = xm1 + (h/6) * (k1x + 2*k2x + 2*k3x + k4x);
    y = ym1 + (h/6) * (k1y + 2*k2y + 2*k3y + k4y);
    
    count = 0;
    iter = 1;
    
    while count < 2
        
        xm2 = xm1;
        ym2 = ym1;
        xm1 = x;
        ym1 = y;
        
        k1x, k1y = odefcn(xm1, ym1);
        k2x, k2y = odefcn(xm1 + k1x * h/2, ym1 + k1y * h/2);
        k3x, k3y = odefcn(xm1 + k2x * h/2, ym1 + k2y * h/2);
        k4x, k4y = odefcn(xm1 + k3x * h  , ym1 + k3y * h  );

        x = xm1 + (h/6) * (k1x + 2*k2x + 2*k3x + k4x);
        y = ym1 + (h/6) * (k1y + 2*k2y + 2*k3y + k4y);
        
        iter = iter + 1;

        if y*ym1 < 0
            count = count+1;
        end
        
    end
    
    xdiff = x0 - (xm2 * (ym1 * y  )   / ((ym2 - ym1) * (ym2 - y  )) +
                + xm1 * (ym2 * y  )   / ((ym1 - ym2) * (ym1 - y  )) +
                + x   * (ym2 * ym1)   / ((y   - ym2) * (y   - ym1)));
    
    return xdiff, iter
    
end

###PARALLEL###

#Grid-stride loop kernel
function gpu_RK4!(points, xdiff, h, iter)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    for i = index:stride:length(points)
        @inbounds xdiff[i], iter[i] = RK4(points[i],0.0, h, iter[i]);
    end
    return nothing
end

function bench_gpu_RK4!(points, xdiff, h, iter)
    kernel = @cuda launch=false gpu_RK4!(points, xdiff, h, iter)
    config = launch_configuration(kernel.fun)
    threads = min(length(points), 1024)
    blocks = ceil(Int, length(points)/1024)


    CUDA.@sync begin
        @cuda threads=threads blocks=blocks gpu_RK4!(points, xdiff, h, iter)
    end
end

h_d = 0.01;
points_d = CuArray{Float32}(0.5:0.0001:20);
xdiff_d = CuArray{Float32}(CUDA.zeros(length(points_d)));
iter_d = CuArray{Float32}(CUDA.zeros(length(points_d)));

using BenchmarkTools
@btime bench_gpu_RK4!($points_d, $xdiff_d, $h_d, $iter_d)

length(points_d)
mean(iter_d)

67.741/length(points_d)/mean(iter_d)
67.741/length(points_d)

###SEQUENTIAL###

function sequential_RK4!(points, xdiff, h, iter)
    for i in eachindex(points, xdiff)
        @inbounds xdiff[i], iter[i] = RK4(points[i], 0.0, h, iter);
    end
    return nothing
end

h_s = 0.00001;
points_s = Float32.(Array(0.5:0.0001:20));
xdiff_s = Float32.(zeros(length(points_s)));
iter_s = Float32.(zeros(length(points_s)));

@btime sequential_RK4!($points_s, $xdiff_s, $h_s, $iter_s)
