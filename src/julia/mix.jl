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
        "root" => "D:\\5-Workspace\\Mix",
        "voice_depot" => "D:\\5-Workspace\\Voice\\",
        "noise_depot" => "D:\\5-Workspace\\GoogleAudioSet",
        "sample_rate" => 16000,
        "voice_level" => [-22.0, -32.0, -42.0],
        "snr" => [20.0, 15.0, 10.0, 5.0, 0.0, -5.0],
        "voice_noise_time_ratio" => 0.1,
        "split_ratio_for_training" => 0.7,
        "training_samples" => 10,
        "test_samples" => 10,
        "seed" => 42,
        "feature" => Dict("frame_length"=>512, "hop_length"=>128, "frame_context"=>11, "nat_frames"=>7),
        "tensor_partition_size(MB)" => 1024,
        "noise_categories" => x
        )
    for i in DATA.list(a["noise_depot"])
        push!(x, Dict("name"=>i,"type"=>"stationary|nonstationary|impulsive","percent"=>0.0))
    end

    #rm(a["mixoutput"], force=true, recursive=true)
    !isdir(a["root"]) && mkpath(a["root"])
    open(joinpath(a["root"],"specification-$(replace(replace("$(now())",":","-"),".","-")).json"),"w") do f
        write(f, JSON.json(a))
    end

    # generate random checksum for level infomation update
    for i in a["noise_categories"]
        p = joinpath(a["noise_depot"],i["name"])
        DATA.touch_checksum(p)
        info("checksum written to $p")
    end
    nothing
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

    lpek = zeros(n)
    lrms = zeros(n)
    lmed = zeros(n)
    leng = zeros(Int64, n)

    # wav must be monochannel and fs==rate
    p = UI.Progress(10)
    for (i,v) in enumerate(a)
        try
            x, fs = WAV.wavread(v)
            assert(fs == typeof(fs)(rate))
            assert(size(x,2) == 1)
            y = view(x,:,1)
            lpek[i] = maximum(abs.(y))
            lrms[i] = FEATURE.rms(y)
            lmed[i] = median(abs.(y))
            leng[i] = length(y)
        catch
            warn(v)
        end
        UI.update(p, i, n)
    end

    index = joinpath(path, "index.level")
    writedlm(index, [a lpek lrms lmed leng], ',')
    info("index build to $index")
    nothing
end





function wavread_safe(path)
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






function wavgen(s, noiselevel, noisesplit, voicelevel, voicesplit; flag="training")

    n_train::Int64 = s["training_samples"]
    n_test::Int64 = s["test_samples"]
    snr::Array{Float64,1} = s["snr"]                  # [-5.0, 0.0, 5.0, 10.0, 15.0, 20.0]
    spl::Array{Float64,1} = s["voice_level"]          # [-22.0, -32.0, -42.0]
    vntr::Float64 = s["voice_noise_time_ratio"]       # 0.1
    split::Float64 = s["split_ratio_for_training"]    # 0.7
    fs::Int64 = s["sample_rate"]
    root::String = s["root"]
    voice::String = s["voice_depot"]
    noise::String = s["noise_depot"]
    noisecat::Array{Dict{String,Any},1} = s["noise_categories"]

    label = Dict{String, Array{Tuple{Int64, Int64},1}}()
    gain = Dict{String, Array{Float64,1}}()
    source = Dict{String, Tuple{String, String}}()

    mixwav = joinpath(root, flag, "wav")
    mkpath(mixwav)

    n = (flag=="training")? n_train:n_test
    n_count = 1

    q = length(voicesplit)
    p = Int64(round(split * q))
    voicesplit_train = view(voicesplit, 1:p)
    voicesplit_test = view(voicesplit, p+1:q)


    for cat in noisecat

        i_n = Int64(round(cat["percent"] * 0.01 * n))

        cname = cat["name"]
        q = length(noisesplit[cname])
        p = Int64(round(split * q))
        noisesplit_train = view(noisesplit[cname], 1:p)
        noisesplit_test = view(noisesplit[cname], p+1:q)

        for j = 1:i_n

                # preparation of randomness
                voice_spl_tt::Float64 = rand(spl)
                snr_tt::Float64 = rand(snr)
                rn_voice::Int64 = (flag=="training")? rand(voicesplit_train):rand(voicesplit_test)
                rn_noise::Int64 = (flag=="training")? rand(noisesplit_train):rand(noisesplit_test)

                # addressing parameters based on generated randomness
                voice_wav::String = voicelevel[rn_voice,1]
                voice_lpk::Float64 = voicelevel[rn_voice,2]
                voice_spl::Float64 = voicelevel[rn_voice,3]
                voice_len::Int64 = voicelevel[rn_voice,4]

                block = noiselevel[cname]
                noise_wav::String = block[rn_noise,1]
                noise_lpk::Float64 = block[rn_noise,2]
                noise_rms::Float64 = block[rn_noise,3]
                noise_med::Float64 = block[rn_noise,4]
                noise_len::Int64 = block[rn_noise,5]

                # record the gains applied to speech and noise
                gain_ = zeros(2)

                # level the random speech to target
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
                gain_[1] = g


                # get the random noise
                x2,fs2 = wavread_safe(noise_wav)
                assert(typeof(fs)(fs2) == fs)
                u = view(x2,:,1)

                # random snr -> calculate noise level based on speech level and snr
                t = 10^((voice_spl_tt-snr_tt)/20)
                noisetype = cat["type"]
                if noisetype == "impulsive"
                    g = t / noise_lpk
                elseif noisetype == "stationary"
                    g = t / noise_rms
                elseif noisetype == "nonstationary"
                    g = t / noise_med
                else
                    error("wrong type in $(i["name"])")
                end
                if g * noise_lpk > 1
                     g = 1 / noise_lpk
                     info("noise avoid clipping $(noise_wav)")
                end
                u .= g .* u
                gain_[2] = g


                # voice-noise time ratio control
                noise_id = replace(relpath(noise_wav,noise), "\\", "+")[1:end-4]
                voice_id = replace(relpath(voice_wav,voice), "\\", "+")[1:end-4]
                # D:\5-Workspace\GoogleAudioSet\Aircraft\m0k5j+_GcfZXqPJf4.wav -> Aircraft + m0k5j+_GcfZXqPJf4
                # D:\4-Workspace\Voice\TIMIT-16k\dr1\fcjf0\sa1.wav  -> TIMIT-16k + dr1 + fcjf0 + sa1
                # D:\5-Workspace\Voice\LP7-16k\CA\CA01_01.wav -> LP7-16k + CA + CA01_01


                pathout = joinpath(mixwav,"$(n_count)+$(noise_id)+$(voice_id)+$(voice_spl_tt)+$(snr_tt).wav")
                gain[pathout] = gain_
                source[pathout] = (voice_wav, noise_wav)
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
                    # case when voice is too short for the noise, extend the voice,
                    # here we don't do cyclic extension with voice,
                    # instead we scatter multiple copies of voice among entire nosie
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
                n_count += 1
            end
            info("[+ $(cname) processed +]")
        end

        label_path = joinpath(root,flag,"label.json")
        gain_path = joinpath(root,flag,"gain.json")
        source_path = joinpath(root,flag,"source.json")
        open(label_path,"w") do f
            write(f, JSON.json(label))
        end
        open(gain_path,"w") do f
            write(f, JSON.json(gain))
        end
        open(source_path,"w") do f
            write(f, JSON.json(source))
        end
        info("label written to $(label_path)")
        info("gain written to $(gain_path)")
        info("source written to $(source_path)")
        nothing
end




# mix procedure implements the specification
function mix(specifijson::String)

    # read the specification for mixing task
    s = JSON.parsefile(specifijson)
    srand(s["seed"])

    fs::Int64 = s["sample_rate"]
    root::String = s["root"]
    voice::String = s["voice_depot"]
    noise::String = s["noise_depot"]
    noisecat::Array{Dict{String,Any},1} = s["noise_categories"]


    # 1. remove existing wav file in the mix folder
    for i in DATA.list(root, t=".wav")
        rm(i, force=true)
    end


    # 2. detect data change and update level information
    !isdir(noise) && error("noise depot doesn't exist")
    sumpercent = 0.0
    for i in noisecat
        catpath = joinpath(noise,i["name"])
        if !DATA.verify_checksum(catpath)
            info("checksum mismatch: updating level index...")
            build_level_index(catpath, fs)
            DATA.update_checksum(catpath)
        end
        sumpercent += i["percent"]
    end
    assert(99.999 < sumpercent < 100.001)

    # 2. noise level indexing format:
    # noiselevel["class-name"][:,1] = path-to-wav
    # noiselevel["class-name"][:,2] = level-peak
    # noiselevel["class-name"][:,3] = level-rms
    # noiselevel["class-name"][:,4] = level-median
    # noiselevel["class-name"][:,5] = length-in-samples
    noiselevel = Dict(x["name"] => readdlm(joinpath(noise, x["name"],"index.level"), ',') for x in noisecat)
    noisesplit = Dict(x["name"] => randperm(size(noiselevel[x["name"]],1)) for x in noisecat)


    # 3. build speech level and check speech integrety
    !isdir(voice) && error("voice depot doesn't exist")
    if !DATA.verify_checksum(voice)
        # build_level_index(s["speech_rootpath"])
        # assume index ready by Matlab:activlevg() and provided as csv in format:
        # path-to-wav, speech-level(dB), length-in-samples
        DATA.update_checksum(voice)
    end

    # 4. speech level indexing format:
    # voicelevel[:,1] = path-to-wav
    # voicelevel[:,2] = peak-level
    # voicelevel[:,3] = speech-level(dB)
    # voicelevel[:,4] = length-in-samples
    voicelevel = readdlm(joinpath(voice,"index.level"), ',', header=false, skipstart=3)
    voicesplit = randperm(size(voicelevel,1))


    # 5. mixing'em up
    wavgen(s, noiselevel, noisesplit, voicelevel, voicesplit)
    wavgen(s, noiselevel, noisesplit, voicelevel, voicesplit, flag="test")
    nothing
end










# remove old feature.h5 and make new
# 87+xxx.wav/mix
# 87+xxx.wav/bm
# bm and mix are matrix of form nfft/2+1-by-frames
function feature(specifijson::String; flag="training")

    # mixed file and label info
    s = JSON.parsefile(specifijson)

    fs::Int64 = s["sample_rate"]
    n_train::Int64 = s["training_samples"]
    n_test::Int64 = s["test_samples"]
    m::Int64 = s["feature"]["frame_length"]
    hp::Int64 = s["feature"]["hop_length"]

    root = s["root"]
    voice = s["voice_depot"]
    noise = s["noise_depot"]

    label = JSON.parsefile(joinpath(root, flag, "label.json"))
    gain = JSON.parsefile(joinpath(root, flag, "gain.json"))
    source = JSON.parsefile(joinpath(root, flag, "source.json"))

    n = (flag=="training")? n_train:n_test
    assert(n == length(label))
    assert(n == length(gain))

    # remove existing .h5 data
    output = joinpath(root, flag, "spectrum.h5")
    rm(output, force=true)
    progress = UI.Progress(10)

    for (i,v) in enumerate(keys(label))

        # p = split(v[1:end-length(".wav")],"+")

        x_mix, fs1 = WAV.wavread(v)
        assert(typeof(fs)(fs1) == fs)
        x_mix = view(x_mix,:,1)

        x_voice,fs2 = WAV.wavread(source[v][1])
        assert(typeof(fs)(fs2) == fs)
        x_voice = view(x_voice,:,1)
        x_voice .*= gain[v][1]

        # x_noise = WAV.wavread(source[v][2])
        # x_noise = view(x_noise,:,1)
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
        srand(s["seed"])
        dither = randn(size(x_purenoise)) * (10^(-120/20))
        x_purenoise .+= dither


        # for verification purpose
        # v_ = v[1:end-length(".wav")]
        # WAV.wavwrite(hcat(x_mix, x_purevoice, x_purenoise), v_*"-decomp.wav",Fs=fs)

        𝕏m, h = STFT2.stft2(x_mix, m, hp, STFT2.sqrthann)
        𝕏v, h = STFT2.stft2(x_purevoice, m, hp, STFT2.sqrthann)
        𝕏n, h = STFT2.stft2(x_purenoise, m, hp, STFT2.sqrthann)
        bm = abs.(𝕏v) ./ (abs.(𝕏v) + abs.(𝕏n))

        HDF5.h5write(output, "$v/bm", bm)
        HDF5.h5write(output, "$v/mix", abs.(𝕏m))
        UI.update(progress, i, n)
    end
    info("feature written to $(output)")
end







# global variance:
# remove old global.h5 and make new
function statistics(specifijson)

    # read the specification for feature extraction
    s = JSON.parsefile(specifijson)
    root = s["root"]
    m = div(s["feature"]["frame_length"], 2) + 1

    pathstat = joinpath(root,"training","stat.h5")
    rm(pathstat, force=true)

    fid = HDF5.h5open(joinpath(root,"training","spectrum.h5"),"r")
    l = length(names(fid))

    # get total frame count of training set
    n = zero(Int128)
    progress = UI.Progress(10)
    for (i,j) in enumerate(names(fid))
        n += size(read(fid[j]["mix"]), 2)
        UI.update(progress, i, l)
    end
    info("global spectrum count(training): $n")

    # get total frame count of validation set
    fidv = HDF5.h5open(joinpath(root,"test","spectrum.h5"),"r")
    lv = length(names(fidv))
    nv = zero(Int128)
    UI.rewind(progress)
    for (i,j) in enumerate(names(fidv))
        nv += size(read(fidv[j]["mix"]), 2)
        UI.update(progress, i, lv)
    end
    info("global spectrum count(validation): $nv")
    close(fidv)


    # get global mean log power spectrum
    μ = zeros(m)
    σ = zeros(m)
    μi = zeros(BigFloat, m, l)

    UI.rewind(progress)
    for (i,j) in enumerate(names(fid))
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
    HDF5.h5write(pathstat, "frames_training", Int64(n))
    HDF5.h5write(pathstat, "frames_validation", Int64(nv))
    assert(n < typemax(Int64))
    info("global results written to $(pathstat)")

    close(fid)
end



function tensorsize_estimate(specifijson)
    
    s = JSON.parsefile(specifijson)
    root = s["root"]
    nfft = s["feature"]["frame_length"]
    nat = s["feature"]["nat_frames"]
    ntxt = s["feature"]["frame_context"]
    limit = s["tensor_partition_size(MB)"]
    assert(isodd(ntxt))

    m = div(nfft, 2) + 1
    r = div(ntxt-1, 2)
    ph = (ntxt+1) * m

    fid = HDF5.h5open(joinpath(root, "training", "spectrum.h5"),"r")
    groups = names(fid)
    index = rand(groups,10)

    nf = zeros(Int64, 10)
    for j = 1:10
        nf[j] = size(read(fid[index[j]]["mix"]), 2)
    end
    data = zeros(ph, sum(nf))
    label = zeros(m, sum(nf))
    (start, fin) = borders(nf)

    for j = 1:10
        tmp = read(fid[index[j]]["mix"])
        data[:, start[j]:fin[j]] = FEATURE.sliding(tmp, r, nat)
        label[:, start[j]:fin[j]] = read(fid[index[j]]["bm"])    
    end

    # =========================output as bin/h5/compressed h5 file===========================

    # pathout = joinpath(tempdir(), "tensor.bin")
    # DATA.writebin(pathout, vcat(Float32.(data), Float32.(label)))

    # option h5:
    pathout = joinpath(tempdir(), "tensor.h5")
    HDF5.h5write(pathout, "data", Float32.(data))
    HDF5.h5write(pathout, "label", Float32.(label))
    
    # option h5 compressed:
    # HDF5.h5open(pathout,"w") do file
    #     file["/"]["data", "shuffle", (), "deflate", 4] = Float32.(data)
    #     file["/"]["label", "shuffle", (), "deflate", 4] = Float32.(label)
    # end

    # =======================================================================================

    # estimate number of bytes per group
    bytes = div(filesize(pathout), 10)
    limit = limit * 1024 * 1024
    ngpp = div(limit, bytes)

    info("bytes per group: $(bytes/1024) KB")
    info("number of groups per partition: $(ngpp)")

    rm(pathout)
    close(fid)
    ngpp
end



# ngpp: number of groups per partition
function tensor(specifijson, ngpp; flag="training")

    s = JSON.parsefile(specifijson)
    root = s["root"]
    nfft = s["feature"]["frame_length"]
    nat = s["feature"]["nat_frames"]
    ntxt = s["feature"]["frame_context"]
    assert(isodd(ntxt))

    tensordir = joinpath(root, flag, "tensor")
    mkpath(tensordir)
    tensorlist = DATA.list(tensordir, t=".h5")
    for i in tensorlist
        rm(i, force=true)
    end

    m = div(nfft, 2) + 1
    r = div(ntxt-1, 2)
    ph = (ntxt+1) * m

    # extract global stat info
    # note that the global stat must be of training
    stat = joinpath(root, "training", "stat.h5")
    # n = h5read(stat,"frames")
    μ = HDF5.h5read(stat,"mu")
    σ = HDF5.h5read(stat,"std")

    # context processing
    fid = HDF5.h5open(joinpath(root, flag, "spectrum.h5"),"r")
    groups = names(fid)


    # gb: group bias
    # np: number of groups to be processed
    function tensorblock(gb, np, k)

        nf = zeros(Int64, np)  # size of each group
        for j = 1:np
            nf[j] = size(read(fid[groups[gb+j]]["mix"]), 2)
        end
        data = zeros(ph, sum(nf))
        label = zeros(m, sum(nf))
        (start, fin) = borders(nf)

        progress = UI.Progress(10)    
        for j = 1:np
            local p = gb+j
            tmp = read(fid[groups[p]]["mix"])
            tmp = (tmp.-μ)./σ
            data[:, start[j]:fin[j]] = FEATURE.sliding(tmp, r, nat)
            label[:, start[j]:fin[j]] = read(fid[groups[p]]["bm"])
            UI.update(progress, j, np)
        end

        # =========================output as bin/h5/compressed h5 file===========================

        # pathout = joinpath(tensordir, "tensor_$k.bin")
        # DATA.writebin(pathout, vcat(Float32.(data), Float32.(label)))
        
        pathout = joinpath(tensordir, "tensor_$k.h5")
        HDF5.h5write(pathout, "data", Float32.(data))
        HDF5.h5write(pathout, "label", Float32.(label))

        # HDF5.h5open(pathout,"w") do file
        #     file["/"]["data", "shuffle", (), "deflate", 4] = Float32.(data)
        #     file["/"]["label", "shuffle", (), "deflate", 4] = Float32.(label)
        # end

        # =======================================================================================
        nothing
    end

    t = 0
    i = 0
    while i+ngpp <= length(groups)
        tensorblock(i, ngpp, t)
        info("partition $t ok")
        i += ngpp
        t += 1
    end
    remain = length(groups)-i
    remain > 0 && (tensorblock(i,remain,t); info("partition $t ok"))
    close(fid)
end








function build(spec)

    mix(spec)
    feature(spec)
    feature(spec, flag="test")
    statistics(spec)
    groupspart = tensorsize_estimate(spec)
    tensor(spec, groupspart)
    tensor(spec, groupspart, flag="test")
end






function process_validset(specification::String, dataset::String, model::String)

    s = JSON.parsefile(specification)   
    root = s["root"]
    nfft = s["feature"]["frame_length"]
    nhp = s["feature"]["hop_length"]
    nat = s["feature"]["nat_frames"]
    ntxt = s["feature"]["frame_context"]
    assert(isodd(ntxt))

    # get global mu and std
    stat = joinpath(root, "training", "stat.h5")
    μ = Float32.(HDF5.h5read(stat, "mu"))
    σ = Float32.(HDF5.h5read(stat, "std"))

    nn = FORWARD.TF{Float32}(model)
    bm = Dict{String, Array{Float32,2}}()

    # load the train/test spectrum+bm dataset
    tid = HDF5.h5open(joinpath(root,"training","spectrum.h5"),"r")
    vid = HDF5.h5open(joinpath(root,"test","spectrum.h5"),"r")

    dset = DATA.list(dataset, t=".wav")
    for i in dset
        bm[i] = FORWARD.vola_processing(nfft, nhp, nat, ntxt, μ, σ, tid, vid, i, nn)
    end
    
    close(vid)
    close(tid)
    
    # log bm error to dataset/../bmerr.h5 
    path5 = joinpath(realpath(joinpath(dataset, "..")), "bmerr.h5")
    HDF5.h5open(path5,"w") do file
        for i in keys(bm)
            # HDF5.h5write(path5, i, bm[i])
            file["/"][i, "shuffle", (), "deflate", 4] = bm[i]
        end
    end
    nothing
end



function benchmark(specification::String, bmerr::String)

    s = JSON.parsefile(specification)   
    m = div(s["feature"]["frame_length"],2)+1

    file = HDF5.h5open(bmerr, "r")
    bm = [(i, mean(read(file[i]),1), mean(read(file[i]),2)) for i in names(file)]
    close(file)

    # bm average over the whole batch
    function gobal_average()
        av = zeros(Float32, m)
        for i in bm
            av .+= vec(i[3])
        end
        av .= av ./ length(bm)
    end

    (gobal_average(),bm)

    # sort!(bm, by=x->sum(x[3]), rev=true)                # worst case by all bins
    # sort!(bm, by=x->sum(view(x[3],13:37,:)), rev=true)  # worst case by bin 13 to 37
    # sort!(bm, by=x->maximum(x[2]), rev=true)            # worst case by highest dm deviation in frames
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
