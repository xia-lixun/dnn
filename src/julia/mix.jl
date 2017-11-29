module MIX
# management of wav files for machine learning projects
# lixun.xia@outlook.com
# 2017-10-16


import WAV
import JSON
import HDF5

include("feature.jl")
include("forward.jl")
include("ui.jl")
include("data.jl")
include("stft2.jl")







#generate template JSON file based on folder contents
function generate_specification()
    
    x = Array{Dict{String,Any},1}()
    a = Dict(
        "build_voice_level_index" => true,
        "build_noise_level_index" => true,

        "mix_root" => "D:\\4-Workspace\\mix",
        "voice_root" => "D:\\4-Workspace\\voice\\TIMIT-16k\\train",
        "noise_root" => "D:\\4-Workspace\\noise\\104Nonspeech-16k",

        "sample_rate" => 16000,
        "voice_level" => [-22.0, -32.0, -42.0],
        "snr" => [20.0, 15.0, 10.0, 5.0, 0.0, -5.0],
        "voice_noise_time_ratio" => 0.1,
        "sample_space"=>17,
        "seed" => 1234,

        "feature" => Dict("frame_length"=>512, "hop_length"=>128, "frame_context"=>11, "nat_frames"=>7),
        "spectrum" => "train.h5",
        "noise_class" => x
        )
    
    for i in DATA.list(a["noise_root"])
        push!(x, Dict("name"=>i,"type"=>"stationary|nonstationary|impulsive","percent"=>0.0))
    end

    #rm(a["mixoutput"], force=true, recursive=true)
    !isdir(a["mix_root"]) && mkpath(a["mix_root"])
    open(joinpath(a["mix_root"],"specification-$(replace(replace("$(now())",":","-"),".","-")).json"),"w") do f
        write(f, JSON.json(a))
    end
end






# calculate index for noise samples:
# peak level: (a) as level of impulsive sounds (b) avoid level clipping in mixing
# rms level: level of stationary sounds
# median level: level of non-stationary sounds
#
# note: wav read errors will be skipped but warning pops up
#       corresponding features remain zero
function build_level_index(path, rate)

    a = DATA.list(path, t=".wav")
    n = length(a)
    lpeak = zeros(n)
    lrms = zeros(n)
    lmedian = zeros(n)
    len = zeros(Int64, n)

    # wav must be monochannel and fs==16000
    p = UI.Progress(10)
    for (i,j) in enumerate(a)
        try
            x, fs = WAV.wavread(j)
            assert(fs == typeof(fs)(rate))
            assert(size(x,2) == 1)
            y = view(x,:,1)
            lpeak[i] = maximum(abs.(y))
            lrms[i] = FEATURE.rms(y)
            lmedian[i] = median(abs.(y))
            len[i] = length(y)
        catch
            warn(j)
        end
        UI.update(p, i, n)
    end

    # save level index to csv
    index = joinpath(path, "index.level")
    writedlm(index, [a lpeak lrms lmedian len], ',')
    info("index build to $index")
end



randi(x::AbstractArray) = rand(1:length(x))

function wavread_safe(path::String)
    try
        x,fs = WAV.wavread(path)
    catch
        #todo: test if my wav class could handle the corner cases of WAV.jl
        #todo: wrap my wav class with c routines to wav.dll, then wrap with julia
        error("missing $path")
    end
end

function cyclic_extend!(xce::AbstractArray, x::AbstractArray)
    n = length(xce)
    for (i,v) in enumerate(Iterators.cycle(x))
        xce[i] = v
        i == n && break
    end
    nothing
end


function borders(partition)
    fin = cumsum(partition)
    beg = vcat(1, 1+fin[1:end-1])
    (beg,fin)
end



# mix procedure implements the specification
function mix(specification_json::String)

    # read the specification for mixing task
    s = JSON.parsefile(specification_json)

    fs::Int64 = s["sample_rate"]                      # 16000
    n::Int64 = s["sample_space"]                      # 17
    snr::Array{Float64,1} = s["snr"]                  # [-5.0, 0.0, 5.0, 10.0, 15.0, 20.0]
    spl::Array{Float64,1} = s["voice_level"]          # [-22.0, -32.0, -42.0]
    vntr::Float64 = s["voice_noise_time_ratio"]       # 0.1

    mixroot::String = s["mix_root"]
    voiceroot::String = s["voice_root"]
    noiseroot::String = s["noise_root"]
    noiseclass::Array{Dict{String,Any},1} = s["noise_class"]

    # remove existing wav file in the mix folder
    for i in DATA.list(mixroot, t=".wav")
        rm(i, force=true)
    end
    mkpath(joinpath(mixroot, "wav"))


    # 1. build noise level and check noise integety
    !isdir(noiseroot) && error("noise root doesn't exist")
    if s["build_noise_level_index"]
        for i in noiseclass
            build_level_index(joinpath(noiseroot,i["name"]), fs)
            DATA.update_checksum(joinpath(noiseroot,i["name"]))
        end
    else
        for i in noiseclass
            !DATA.verify_checksum(joinpath(noiseroot,i["name"])) && error("checksum $(i["name"])")
            info("noise checksum pass")
        end
    end

    # 2. noise level indexing format: 
    # nli["class-name"][:,1] = path-to-wav
    # nli["class-name"][:,2] = level-peak
    # nli["class-name"][:,3] = level-rms
    # nli["class-name"][:,4] = level-median
    # nli["class-name"][:,5] = length-in-samples
    nli = Dict( x["name"] => readdlm(joinpath(noiseroot, x["name"],"index.level"), ',') for x in noiseclass)


    # 3. build speech level and check speech integrety
    !isdir(voiceroot) && error("voice root doesn't exist")
    if s["build_voice_level_index"]
        # build_level_index(s["speech_rootpath"])
        # assume index ready by Matlab:activlevg() and provided as csv in format: 
        # path-to-wav, speech-level(dB), length-in-samples
        DATA.update_checksum(voiceroot)
    else
        !DATA.verify_checksum(voiceroot) && error("voice data checksum error")
        info("voice checksum pass")
    end

    # 4. speech level indexing format: 
    # sli[:,1] = path-to-wav
    # sli[:,2] = peak-level
    # sli[:,3] = speech-level(dB)
    # sli[:,4] = length-in-samples
    sli = readdlm(joinpath(voiceroot,"index.level"), ',', header=false, skipstart=3)



    # 5. mixing'em up
    mixcount = 1
    label = Dict{String, Array{Tuple{Int64, Int64},1}}()
    gain = Dict{String, Array{Float64,1}}()
    srand(s["seed"])

    # dn: dictionary of noise class
    for dn in noiseclass
        span = Int64(round(dn["percent"] * 0.01 * n))
        for isp = 1:span
            
            # preparation of randomness
            voice_spl_tt::Float64 = spl[randi(spl)]
            snr_tt::Float64 = snr[randi(snr)]
            rns::Int64 = randi(view(sli,:,2))
            rnn::Int64 = randi(view(nli[dn["name"]],:,2))

            # addressing parameters based on generated randomness
            voice_wav::String = sli[rns,1]
            voice_lpk::Float64 = sli[rns,2]
            voice_spl::Float64 = sli[rns,3]
            voice_len::Int64 = sli[rns,4]

            noise_wav::String = nli[dn["name"]][rnn,1]
            noise_lpk::Float64 = nli[dn["name"]][rnn,2]
            noise_rms::Float64 = nli[dn["name"]][rnn,3]
            noise_med::Float64 = nli[dn["name"]][rnn,4]
            noise_len::Int64 = nli[dn["name"]][rnn,5]

            gvec = zeros(2)

            # level the random speech to target level
            x1,fs1 = wavread_safe(voice_wav)
            assert(typeof(fs)(fs1) == fs)
            x = view(x1,:,1)

            g = 10^((voice_spl_tt-voice_spl)/20)
            if g * voice_lpk > 1 
                g = 1 / voice_lpk
                voice_spl_tt = voice_spl + 20log10(g+eps())
                info("voice avoid clipping $(voice_wav):$(voice_spl)->$(voice_spl_tt) dB")
            end
            x .= g .* x
            gvec[1] = g
            

            # get the random noise data
            x2,fs2 = wavread_safe(noise_wav)
            assert(typeof(fs)(fs1) == fs)
            u = view(x2,:,1)

            # random snr -> calculate noise level based on speech level and snr
            t = 10^((voice_spl_tt-snr_tt)/20)
            noisetype = dn["type"]
            if noisetype == "impulsive" 
                g = t / noise_lpk
            elseif noisetype == "stationary"
                g = t / noise_rms
            elseif noisetype == "nonstationary"
                g = t / noise_med
            else
                error("wrong type in $(dn["name"])")
            end
            if g * noise_lpk > 1
                 g = 1 / noise_lpk
                 info("noise avoid clipping $(noise_wav)")
            end
            u .= g .* u
            gvec[2] = g
            

            # voice-noise time ratio control
            noise_id = replace(relpath(noise_wav,noiseroot), "\\", "+")[1:end-4]
            voice_id = replace(relpath(voice_wav,voiceroot), "\\", "+")[1:end-4]
            # D:\4-Workspace\noise\104Nonspeech-16k\impulsive\n48.wav -> impulsive+n48
            # D:\4-Workspace\voice\TIMIT-16k\train\dr1\fcjf0\sa1.wav  -> dr1+fcjf0+sa1
            
            
            pathout = joinpath(mixroot,"wav","$(mixcount)+$(noise_id)+$(voice_id)+$(voice_spl_tt)+$(snr_tt).wav")
            gain[pathout] = gvec
            η = voice_len/noise_len

            if η > vntr
                # case when voice is too long for the noise, extend the noise in cyclic
                noise_len_extend = Int64(round(voice_len / vntr))
                u_extend = zeros(noise_len_extend)
                cyclic_extend!(u_extend, u)
                
                r = rand(1:noise_len_extend-voice_len)
                u_extend[r:r+voice_len-1] += x
                WAV.wavwrite(u_extend, pathout, Fs=fs)
                label[pathout] = [(r, r+voice_len-1)]

            elseif η < vntr
                # case when voice is too short for the noise, extend the voice, here we don't do cyclic extension
                # with voice, instead we scatter multiple copies of voice among entire nosie
                voice_len_tt = Int64(round(noise_len * vntr))
                λ = voice_len_tt / voice_len   # 3.3|3.0
                λr = floor(λ)                  # 3.0|3.0
                λ1 = λr - 1.0                  # 2.0|2.0
                λ2 = λ - λr + 1.0              # 1.3|1.0
                
                voice_len_extend = Int64(round(voice_len * λ2))
                x_extend = zeros(voice_len_extend)
                cyclic_extend!(x_extend, x)
                # obs! length(x_extended) >= voice_len
                
                ζ = Int64(round(noise_len / λ))
                partition = zeros(Int64, Int64(λ1)+1)
                for i = 1:Int64(λ1)
                    partition[i] = ζ
                end
                partition[end] = noise_len - Int64(λ1) * ζ
                assert(partition[end] >= ζ)
                shuffle!(partition)
                (beg,fin) = borders(partition)
                
                labelmark = Array{Tuple{Int64, Int64},1}()                
                for (i,v) in enumerate(partition)
                    if v > ζ
                        r = rand(beg[i] : fin[i]-voice_len_extend)
                        u[r:r+voice_len_extend-1] += x_extend
                        push!(labelmark,(r,r+voice_len_extend-1))
                    else
                        r = rand(beg[i] : fin[i]-voice_len)
                        u[r:r+voice_len-1] += x
                        push!(labelmark,(r,r+voice_len-1))
                    end
                end
                
                WAV.wavwrite(u, pathout, Fs=fs)
                label[pathout] = labelmark

            else
                # this is a rare case as usually you don't encounter precise floating point value equals eta
                # if so probably something's wrong
                r = rand(1:noise_len-voice_len)
                u[r:r+voice_len-1] += x
                WAV.wavwrite(u, pathout, Fs=fs)
                label[pathout] = [(r, r+voice_len-1)]
            end
            mixcount += 1
        end
        info("[+ $(dn["name"]) processed +]")
    end

    open(joinpath(mixroot,"label.json"),"w") do f
        write(f, JSON.json(label))
    end
    open(joinpath(mixroot,"gain.json"),"w") do f
        write(f, JSON.json(gain))
    end    
    info("label written to $(joinpath(mixroot,"label.json"))")
    info("gain written to $(joinpath(mixroot,"gain.json"))")
end







# remove old feature.h5 and make new
function feature(specification_json::String, label_json::String, gain_json::String)

    # mixed file and label info
    label = JSON.parsefile(label_json)
    gain = JSON.parsefile(gain_json)
    sptn= JSON.parsefile(specification_json)

    sr = sptn["sample_rate"]
    n = sptn["sample_space"]
    m = sptn["feature"]["frame_length"]
    hp = sptn["feature"]["hop_length"]

    mixroot = sptn["mix_root"]
    voiceroot = sptn["voice_root"]
    noiseroot = sptn["noise_root"]
    spectrum = sptn["spectrum"]

    assert(n == length(label))
    assert(n == length(gain))
    
    # remove existing .h5 training/valid/test data
    output = joinpath(mixroot, spectrum)
    rm(output, force=true)
    progress = UI.Progress(10)

    for (i,v) in enumerate(keys(label))

        p = split(v[1:end-length(".wav")],"+")
        # [1]"D:\\mix-utility\\mixed\\1"
        # [2]"impulsive"
        # [3]"n48"
        # [4]"dr1"      
        # [5]"mklw0"    
        # [6]"sa1"
        # [7]"-22.0"    
        # [8]"20.0"

        x_mix, fs = WAV.wavread(v)
        assert(typeof(sr)(fs) == sr)
        x_mix = view(x_mix,:,1)

        x_voice,fs = WAV.wavread(joinpath(voiceroot,p[4],p[5],p[6]) * ".wav")
        assert(typeof(sr)(fs) == sr)
        x_voice = view(x_voice,:,1)
        x_voice .*= gain[v][1]

        # x_noise = WAV.wavread(joinpath(noiseroot,p[2],p[3]) * ".wav")[1][:,1]
        # x_noise .*= gain[v][2]

        # retrive pure voice
        x_purevoice= zeros(size(x_mix))
        for k in label[v]
            if k[2]-k[1]+1 == length(x_voice)
                x_purevoice[k[1]:k[2]] = x_voice
            else
                cyclic_extend!(view(x_purevoice,k[1]:k[2]), x_voice)
            end
        end
        
        # retrieve pure noise and add dithering
        x_purenoise = x_mix - x_purevoice
        srand(sptn["seed"])
        dither = randn(size(x_purenoise)) * (10^(-120/20))
        x_purenoise .+= dither


        # for verification purpose        
        # v_ = v[1:end-length(".wav")]
        # WAV.wavwrite(hcat(x_mix, x_purevoice, x_purenoise), v_*"-decomp.wav",Fs=sr)

        𝕏m, h = STFT2.stft2(x_mix, m, hp, STFT2.sqrthann)
        𝕏v, h = STFT2.stft2(x_purevoice, m, hp, STFT2.sqrthann)
        𝕏n, h = STFT2.stft2(x_purenoise, m, hp, STFT2.sqrthann)
    
        bm = abs.(𝕏v) ./ (abs.(𝕏v) + abs.(𝕏n))
        
        HDF5.h5write(output, "$v/bm", bm)
        HDF5.h5write(output, "$v/mix", log.(abs.(𝕏m).+eps()))

        UI.update(progress, i, n)
    end
    info("feature written to $(output)")
end







# global variance:
# remove old global.h5 and make new
function statistics(specification)

    # read the specification for feature extraction
    s = JSON.parsefile(specification)
    mixroot = s["mix_root"]
    m = div(s["feature"]["frame_length"], 2) + 1

    pathstat = joinpath(mixroot,"global.h5")
    rm(pathstat, force=true)
    
    fid = HDF5.h5open(joinpath(mixroot,s["spectrum"]),"r")
    l = length(names(fid))

    # get global frame count
    n = zero(Int128)                            
    progress = UI.Progress(10)
    for (i,j) in enumerate(names(fid))   
        n += size(read(fid[j]["mix"]), 2)        
        UI.update(progress, i, l)
    end
    info("global spectrum count: $n")

    # get global mean log power spectrum
    μ = zeros(m)
    σ = zeros(m)
    μi = zeros(BigFloat, m, l)

    UI.rewind(progress)
    for(i,j) in enumerate(names(fid))
        x = read(fid[j]["mix"])                 
        for k = 1:m
            μi[k,i] = sum_kbn(view(x,k,:))
        end
        UI.update(progress, i, l)
    end
    for k = 1:m
        μ[k] = sum_kbn(view(μi,k,:))/n
    end
    info("global spectrum μ (dimentionless): $(mean(μ))")

    # get global std for unit variance
    fill!(μi, zero(BigFloat))
    UI.rewind(progress)
    for(i,j) in enumerate(names(fid))
        x = read(fid[j]["mix"])              
        for k = 1:m
            μi[k,i] = sum_kbn((view(x,k,:)-μ[k]).^2)
        end
        UI.update(progress, i, l)
    end
    for k = 1:m
        σ[k] = sqrt(sum_kbn(view(μi,k,:))/(n-1))
    end
    info("global spectrum σ (dimentionless): $(mean(σ))")

    HDF5.h5write(pathstat, "mu", μ)
    HDF5.h5write(pathstat, "std", σ)
    HDF5.h5write(pathstat, "frames", Int64(n))
    assert(n < typemax(Int64))
    info("global results written to $(pathstat)")

    close(fid)
end




#######line of pure graceful and joy#######
# specification.json
# partitions: number of .h5's as output
function tensor(specification, partitions::Int64, specification_train)

    # read the specification for feature extraction
    s = JSON.parsefile(specification)
    
    mixroot = s["mix_root"]
    nfft = s["feature"]["frame_length"]
    nat = s["feature"]["nat_frames"]
    ntxt = s["feature"]["frame_context"]
    assert(isodd(ntxt))

    m = div(nfft, 2) + 1
    r = div(ntxt-1, 2)
    ph = (ntxt+1) * m

    # extract global stat info
    # note that the global stat must be of training
    stat = joinpath(JSON.parsefile(specification_train)["mix_root"], "global.h5")
    # n = h5read(stat,"frames")
    μ = HDF5.h5read(stat,"mu")
    σ = HDF5.h5read(stat,"std")

    # context processing
    fid = HDF5.h5open(joinpath(mixroot, s["spectrum"]),"r")
    groups = names(fid)
    np = div(length(groups), partitions)  # groups per partitions

    
    for i = 0:partitions-1

        # find out the size of each partition
        nf = zeros(Int64, np)
        for j = 1:np
            nf[j] = size(read(fid[groups[i*np+j]]["mix"]), 2)
        end  
        data = zeros(ph, sum(nf))
        label = zeros(m, sum(nf))

        # fill in each partition with context data and nat data
        (start, fin) = borders(nf)
        progress = UI.Progress(10)
        for j = 1:np

            tmp = read(fid[groups[i*np+j]]["mix"])
            tmp = (tmp.-μ)./σ
            data[:, start[j]:fin[j]] = FEATURE.sliding(tmp, r, nat)
            label[:, start[j]:fin[j]] = read(fid[groups[i*np+j]]["bm"])

            UI.update(progress, j, np)
        end

        pathout = joinpath(mixroot, "tensor-$i.h5")
        HDF5.h5write(pathout, "data", Float32.(data))
        HDF5.h5write(pathout, "label", Float32.(label))
        info("partition $i ok")
    end

    close(fid)
end








function mixup()

    valid_spec = "D:\\4-Workspace\\mix\\valid\\specification-2017-11-22T21-30-28-642.json"
    valid_lab = "D:\\4-Workspace\\mix\\valid\\label.json"
    valid_gain = "D:\\4-Workspace\\mix\\valid\\gain.json"

    train_spec = "D:\\4-Workspace\\mix\\train\\specification-2017-11-22T21-30-28-642.json"
    train_lab = "D:\\4-Workspace\\mix\\train\\label.json"
    train_gain = "D:\\4-Workspace\\mix\\train\\gain.json"

    # mix(train_spec)                                  # generate mixed wav with labelings and gains
    # feature(train_spec, train_lab, train_gain)       # extract plain features, to valid.h5/train.h5
    # statistics(train_spec)                                # find out the global stats: mean/std/total frames
    # tensor(train_spec, 50, train_spec)                           # convert plain features to tensor input    

    mix(valid_spec)                                  # generate mixed wav with labelings and gains
    feature(valid_spec, valid_lab, valid_gain)       # extract plain features, to valid.h5/train.h5
    statistics(valid_spec)                                # find out the global stats: mean/std/total frames
    tensor(valid_spec, 25, train_spec)                           # convert plain features to tensor input
end




function process_dataset(specification::String, dataset::String; model::String = "")

    dset = DATA.list(dataset, t=".wav")
    pr = UI.Progress(10)
    n = length(dset)
    for (i,j) in enumerate(dset)
        FORWARD.vola_processing(specification, j, model=model)
        UI.update(pr, i, n)
    end
end



# module
end













            # if mr[1] <= η <= mr[2]
            #     rd = rand(1:q-p)
            #     u[rd:rd+p-1] += x
            #     # clipping sample if over-range?
            #     path = joinpath(s["mix_root"],"wav","$(fcount)+$(nid)+$(sid)+1+1+$(sp)+$(sn).wav")
            #     WAV.wavwrite(u, path, Fs=fs)
            #     label[path] = [(rd, rd+p-1)]
            #     gain[path] = gvec
            # # η > mr[2] or η < mr[1]    
            # else 
            #     np = 1
            #     nq = 1
            #     while !(mr[1] <= η <= mr[2])
            #         η > mr[2] && (nq += 1)
            #         η < mr[1] && (np += 1)
            #         η = (np*p)/(nq*q)                    
            #     end
            #     path = joinpath(s["mix_root"],"wav","$(fcount)+$(nid)+$(sid)+$(np)+$(nq)+$(sp)+$(sn).wav")
            #     stamp = Array{Tuple{Int64, Int64},1}()

            #     u = repeat(u, outer=nq)
            #     pp = Int64(floor((nq*q)/np)) 
            #     for k = 0:np-1
            #         rd = k*pp+rand(1:pp-p)
            #         u[rd:rd+p-1] += x
            #         push!(stamp,(rd, rd+p-1))
            #     end
            #     WAV.wavwrite(u, path, Fs=fs)
            #     label[path] = stamp
            #     gain[path] = gvec
            # end