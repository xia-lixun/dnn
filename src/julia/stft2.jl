module STFT2



import WAV
include("feature.jl")




# filter bank with square-root hann window for hard/soft masking
function stft2(x::AbstractArray{T,1}, sz::Int64, hp::Int64, wn) where T <: AbstractFloat
# short-time fourier transform
# input:
#     x    input time series
#     sz   size of the fft
#     hp   hop size in samples
#     wn   window to use
#     sr   sample rate
# output:
#     𝕏    complex STFT output (DC to Nyquist)
#     h    unpacked sample length of the signal in time domain
p = FEATURE.Frame1D{Int64}(0, sz, hp, 0)
𝕏,h = FEATURE.spectrogram(x, p, sz, window=wn, zero_init=true)
𝕏,h
end



function stft2(𝕏::AbstractArray{Complex{T},2}, h::Int64, sz::Int64, hp::Int64, wn) where T <: AbstractFloat
    # input:
    #    𝕏   complex spectrogram (DC to Nyquist)
    #    h   unpacked sample length of the signal in time domain
    # output time series reconstructed
    𝕎 = wn(T,sz) ./ (T(sz/hp))
    𝕏 = vcat(𝕏, conj!(𝕏[end-1:-1:2,:]))
    𝕏 = real(ifft(𝕏,1)) .* 𝕎

    y = zeros(T,h)
    n = size(𝕏,2)
    for k = 0:n-1
        y[k*hp+1 : k*hp+sz] .+= 𝕏[:,k+1]
    end
    y
end



sqrthann(T,n) = sqrt.(FEATURE.hann(T,n,flag="periodic"))


function idealsoftmask()
    
    x1,fs = WAV.wavread("D:\\Git\\dnn\\stft_example\\sound001.wav")
    x2,fs = WAV.wavread("D:\\Git\\dnn\\stft_example\\sound002.wav")
    x1 = view(x1,:,1)
    x2 = view(x2,:,1)

    M = min(length(x1), length(x2))
    x1 = view(x1,1:M)
    x2 = view(x2,1:M)
    x = x1 + x2

    nfft = 1024
    hp = div(nfft,4)

    pmix, h0 = stft2(x, nfft, hp, sqrthann)
    px1, h1 = stft2(x1, nfft, hp, sqrthann)
    px2, h2 = stft2(x2, nfft, hp, sqrthann)

    bm = abs.(px1) ./ (abs.(px1) + abs.(px2))
    py1 = bm .* pmix
    py2 = (1-bm) .* pmix

    scale = 2
    y = stft2(pmix, h0, nfft, hp, sqrthann) * scale
    y1 = stft2(py1, h0, nfft, hp, sqrthann) * scale
    y2 = stft2(py2, h0, nfft, hp, sqrthann) * scale

    y = view(y,1:M)
    y1 = view(y1,1:M)
    y2 = view(y2,1:M)

    delta = 10log10(sum(abs.(x-y).^2)/sum(x.^2))
    bm,y1,y2
    #histogram(bm[100,:])
end




## module    
end