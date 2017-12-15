module FEATURE




AWEIGHT_48kHz_BA = [0.234301792299513 -0.468603584599025 -0.234301792299515 0.937207169198055 -0.234301792299515 -0.468603584599025 0.234301792299512;
                    1.000000000000000 -4.113043408775872 6.553121752655049 -4.990849294163383 1.785737302937575 -0.246190595319488 0.011224250033231]'

AWEIGHT_16kHz_BA = [0.531489829823557 -1.062979659647115 -0.531489829823556 2.125959319294230 -0.531489829823558 -1.062979659647116 0.531489829823559;
                    1.000000000000000 -2.867832572992163  2.221144410202311 0.455268334788664 -0.983386863616284 0.055929941424134 0.118878103828561]'

                    
    
# transfer function filter in z-domain
function tf_filter(B, A, x)
    #   y(n)        b(1) + b(2)Z^(-1) + ... + b(M+1)Z^(-M)
    # --------- = ------------------------------------------
    #   x(n)        a(1) + a(2)Z^(-1) + ... + a(N+1)Z^(-N)
    #
    #   y(n)a(1) = x(n)b(1) + b(2)x(n-1) + ... + b(M+1)x(n-M)
    #              - a(2)y(n-1) - a(3)y(n-2) - ... - a(N+1)y(n-N)
    #
    if A[1] != 1.0
        B = B / A[1]
        A = A / A[1]
    end
    M = length(B)-1
    N = length(A)-1
    Br = flipdim(B,1)
    As = A[2:end]
    L = size(x,2)

    y = zeros(size(x))
    x = [zeros(M,L); x]
    s = zeros(N,L)

    for j = 1:L
        for i = M+1:size(x,1)
            y[i-M,j] = dot(Br, x[i-M:i,j]) - dot(As, s[:,j])
            s[2:end,j] = s[1:end-1,j]
            s[1,j] = y[i-M,j] 
        end
    end
    y
end





# n_update  (shift samples)
# n_overlap (overlapping samples)
# n_block   (block size)

function hamming(T, n; flag="")

    lowercase(flag) == "periodic" && (n += 1)
    ω = Array{T,1}(n)
    α = T(0.54)
    β = 1 - α
    for i = 0:n-1
        ω[i+1] = α - β * T(cos(2π * i / (n-1)))
    end
    lowercase(flag) == "periodic" && (return ω[1:end-1])
    ω
end

function hann(T, n; flag="")

    lowercase(flag) == "periodic" && (n += 1)
    ω = Array{T,1}(n)
    α = T(0.5)
    β = 1 - α
    for i = 0:n-1
        ω[i+1] = α - β * T(cos(2π * i / (n-1)))
    end
    lowercase(flag) == "periodic" && (return ω[1:end-1])
    ω
end




# immutable type definition
# note that BlockProcessing{Int16}(1024.0, 256.0, 0) is perfectly legal as new() will convert every parameter to T
# but BlockProcessing{Int16}(1024.0, 256.3, 0) would not work as it raises InexactError()
# also note that there is not white space between BlockProcessing and {T <: Integer}
struct Frame1D{T <: Integer}
    rate::T
    block::T
    update::T
    overlap::T
    Frame1D{T}(r, x, y, z) where {T <: Integer} = x < y ? error("block size must ≥ update size!") : new(r, x, y, x-y)
end
# we define an outer constructor as the inner constructor infers the overlap parameter
# again the block_len and update_len accepts Integers as well as AbstructFloat with no fractions
#
# example type outer constructors: 
# FrameInSample(fs, block, update) = Frame1D{Int64}(fs, block, update, 0)
# FrameInSecond(fs, block, update) = Frame1D{Int64}(fs, floor(block * fs), floor(update * fs), 0)


# extend array x with prefix/appending zeros for frame slicing
# 1. this is an utility function used by get_frames()
# 2. new data are allocated, so origianl x is untouched.
# 3. zero_init = true: the first frame will have zeros of length nfft-nhop
# 4. zero_append = true: the last frame will partially contain data of original x
function tile(x::AbstractArray{T,1}, p::Frame1D{U}; zero_init=false, zero_append=false) where {T <: AbstractFloat, U <: Integer}
    
    zero_init && (x = [zeros(T, p.overlap); x])                                     # zero padding to the front for defined init state
    length(x) < p.block && error("signal length must be at least one block!")       # detect if length of x is less than block size
    n = div(size(x,1) - p.block, p.update) + 1                                      # total number of frames to be processed
    
    if zero_append
        m = rem(size(x,1) - p.block, p.update)
        if m != 0
            x = [x; zeros(T, p.update-m)]
            n += 1
        end
    end
    (x,n)
end



# function    : get_frames
# x           : array of AbstractFloat {Float64, Float32, Float16, BigFloat}
# p           : frame size immutable struct
# zero_init   : simulate the case when block buffer is init to zero and the first update comes in
# zero_append : simulate the case when remaining samples of x doesn't make up an update length
# 
# example:
# x = collect(1.0:100.0)
# p = Frame1D{Int64}(8000, 17, 7.0, 0)
# y,h = get_frames(x, p) where h is the unfold length in time domain
function get_frames(x::AbstractArray{T,1}, p::Frame1D{U}; window=ones, zero_init=false, zero_append=false) where {T <: AbstractFloat, U <: Integer}
    
    x, n = tile(x, p, zero_init = zero_init, zero_append = zero_append)
    
    ω = window(T, p.block)
    y = zeros(T, p.block, n)
    h = 0
    for i = 1:n
        y[:,i] = ω .* x[h+1:h+p.block]
        h += p.update
    end
    # h is the total hopping size, +(p.block - p.update) for total non-overlapping length
    (y,h+(p.block-p.update))
end

# example:
# x = collect(1.0:100.0)
# p = Frame1D{Int64}(8000, 17, 7.0, 0)
# y,h = spectrogram(x, p, nfft, window=hamming, zero_init=true, zero_append=true) where h is the unfold length in time domain
function spectrogram(x::AbstractArray{T,1}, p::Frame1D{U}, nfft::U; window=ones, zero_init=false, zero_append=false) where {T <: AbstractFloat, U <: Integer}
    
    nfft < p.block && error("nfft length must be greater than or equal to block/frame length")
    x, n = tile(x, p, zero_init = zero_init, zero_append = zero_append)
    m = div(nfft,2)+1

    ω = window(T, nfft)
    P = plan_fft(ω)
    𝕏 = zeros(Complex{T}, m, n)
    h = 0
    for i = 1:n
        ξ = P * ( ω .* [x[h+1:h+p.block]; zeros(T,nfft-p.block)] )
        𝕏[:,i] = ξ[1:m]
        h += p.update
    end
    (𝕏,h+(p.block-p.update))
end




# v: indicates vector <: AbstractFloat
energy(v) = x.^2
intensity(v) = abs.(v)
zero_crossing_rate(v) = floor.((abs.(diff(sign.(v)))) ./ 2)

function short_term(f, x::AbstractArray{T,1}, p::Frame1D{U}; zero_init=false, zero_append=false) where {T <: AbstractFloat, U <: Integer}
    frames, lu = get_frames(x, p, zero_init=zero_init, zero_append=zero_append)
    n = size(frames,2)
    ste = zeros(T, n)
    for i = 1:n
        ste[i] = sum_kbn(f(frames[:,i])) 
    end
    ste
end

pp_norm(v) = (v - minimum(v)) ./ (maximum(v) - minimum(v))
stand(v) = (v - mean(v)) ./ std(v)
hz_to_mel(hz) = 2595 * log10.(1 + hz * 1.0 / 700)
mel_to_hz(mel) = 700 * (10 .^ (mel * 1.0 / 2595) - 1)


# calculate power spectrum of 1-D array on a frame basis
# note that T=Float16 may not be well supported by FFTW backend
function power_spectrum(x::AbstractArray{T,1}, p::Frame1D{U}, nfft::U; window=ones, zero_init=false, zero_append=false) where {T <: AbstractFloat, U <: Integer}
    
    nfft < p.block && error("nfft length must be greater than or equal to block/frame length")
    x, n = tile(x, p, zero_init = zero_init, zero_append = zero_append)

    ω = window(T, nfft)
    f = plan_fft(ω)
    m = div(nfft,2)+1
    ℙ = zeros(T, m, n)
    ρ = T(1 / nfft)

    h = 0
    for i = 1:n
        ξ = f * (ω .* [x[h+1:h+p.block]; zeros(T,nfft-p.block)]) # typeof(ξ) == Array{Complex{T},1} 
        ℙ[:,i] = ρ * ((abs.(ξ[1:m])).^2)
        h += p.update
    end
    (ℙ,h+(p.block-p.update))
end



# calculate filter banks
function filter_banks(T, rate::U, nfft::U; filt_num=26, fl=0, fh=div(rate,2)) where {U <: Integer}
    
    fh > div(rate,2) && error("high frequency must be less than or equal to nyquist frequency!")
    
    ml = hz_to_mel(fl)
    mh = hz_to_mel(fh)
    mel_points = linspace(ml, mh, filt_num+2)
    hz_points = mel_to_hz(mel_points)

    # round frequencies to nearest fft bins
    𝕓 = U.(floor.((hz_points/rate) * (nfft+1)))
    #print(𝕓)

    # first filterbank will start at the first point, reach its peak at the second point
    # then return to zero at the 3rd point. The second filterbank will start at the 2nd
    # point, reach its max at the 3rd, then be zero at the 4th etc.
    𝔽 = zeros(T, filt_num, div(nfft,2)+1)

    for i = 1:filt_num
        for j = 𝕓[i]:𝕓[i+1]
            𝔽[i,j+1] = T((j - 𝕓[i]) / (𝕓[i+1] - 𝕓[i]))
        end
        for j = 𝕓[i+1]:𝕓[i+2]
            𝔽[i,j+1] = T((𝕓[i+2] - j) / (𝕓[i+2] - 𝕓[i+1]))
        end
    end
    𝔽
end


function filter_bank_energy(x::AbstractArray{T,1}, p::Frame1D{U}, nfft::U; window=ones, zero_init=false, zero_append=false, filt_num=26, fl=0, fh=div(p.rate,2), use_log=false) where {T <: AbstractFloat, U <: Integer}

    ℙ = power_spectrum(x, p, nfft, window=window, zero_init=zero_init, zero_append=zero_append)
    𝔽 = filter_banks(T, p.rate, nfft, filt_num=filt_num, fl=fl, fh=fh)
    ℙ = 𝔽 * ℙ
    use_log && (log.(ℙ))
    ℙ
end



# T could be AbstractFloat for best performance
# but defined as Real for completeness.
function local_maxima(x::AbstractArray{T,1}) where {T <: Real}
    
    gtl = [false; x[2:end] .> x[1:end-1]]
    gtu = [x[1:end-1] .>= x[2:end]; false]
    imax = gtl .& gtu
    # return as BitArray mask of true or false
end






# Get frame context from spectrogram x with radius r
# 1. x must be col major, i.e. each col is a spectrum frame for example, 257 x L matrix
# 2. y will be (257*(neighbour*2+1+nat)) x L
# 3. todo: remove allocations for better performance
symm(i,r) = i-r:i+r

# r: radius
# t: noise estimation frames
function sliding(x::Array{T,2}, r::Int64, t::Int64) where T <: AbstractFloat

    m, n = size(x)
    head = repmat(x[:,1], 1, r)
    tail = repmat(x[:,end], 1, r)
    x = hcat(head, x, tail)
    y = zeros(T, (2r+2)*m, n)

    for i = 1:n
        focus = view(x,:,symm(r+i,r))
        nat = sum(view(focus,:,1:t), 2) / t
        y[:,i] = vec(hcat(focus,nat))
    end
    y
end



sigmoid(x::T) where T <: AbstractFloat = one(T) / (one(T) + exp(-x))
rms(x,dim) = sqrt.(sum((x.-mean(x,dim)).^2,dim)/size(x,dim))
rms(x) = sqrt(sum((x-mean(x)).^2)/length(x))





# module
end