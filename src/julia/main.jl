
include("mix.jl")




function build(path_specification)

    s = Mix.Specification(path_specification)
    !isdir(s.root_noise) && error("noise depot doesn't exist")
    !isdir(s.root_speech) && error("speech depot doesn't exist")
    mkpath(s.root_mix)

    for i in s.noise_groups
        path = joinpath(s.root_noise,i["name"])
        if !Mix.FileSystem.verify_checksum(path)
            info("checksum mismatch [$(i["name"])], updating level statistics")
            Mix.build_level_json(path, s.sample_rate)
            Mix.FileSystem.update_checksum(path)
        end
    end
    if !Mix.FileSystem.verify_checksum(s.root_speech)
        info("[todo - calculate speech level online]")
        Mix.FileSystem.update_checksum(s.root_speech)
    end
    Mix.Fast.rand_stabilizer(s.seed, 10^6)
    data = Mix.Layout(s)
    info("data layout formed")

    pid = addprocs(2)
    remotecall_fetch(include, pid[1], "mix.jl")
    remotecall_fetch(include, pid[2], "mix.jl")
    info("parallel environments loaded")


    fi = remotecall(Mix.Fast.rand_stabilizer, pid[1], s.seed-1, 10^6)
    ft = remotecall(Mix.Fast.rand_stabilizer, pid[2], s.seed+1, 10^6)
    fetch(fi)
    fetch(ft)


    fi = remotecall(Mix.wavgen, pid[1], s, data)
    ft = remotecall(Mix.wavgen, pid[2], s, data, flag="test")
    inft = fetch(ft)
    infi = fetch(fi)
    info("time series mixed")
    

    fi = remotecall(Mix.feature, pid[1], s, infi)
    ft = remotecall(Mix.feature, pid[2], s, inft, flag="test")
    fetch(ft)
    fetch(fi)
    info("feature extracted")


    fi = remotecall(Mix.statistics, pid[1], s)
    ft = remotecall(Mix.statistics, pid[2], s, flag="test")
    stst = fetch(ft)
    stsi = fetch(fi)
    info("total frames  [train,test] = [$(stsi["frames"]),$(stst["frames"])]")
    info("mean spectrum [train,test] = [$(mean(stsi["mu_spectrum"])), $(mean(stst["mu_spectrum"]))]")
    info("mean ratiomsk [train,test] = [$(mean(stsi["mu_ratiomask"])), $(mean(stst["mu_ratiomask"]))]")


    fi = remotecall(Mix.sdr_benchmark, pid[1], joinpath(s.root_mix, "test", "decomposition"), joinpath(s.root_mix, "test", "oracle", "dft"))
    ft = remotecall(Mix.sdr_benchmark, pid[2], joinpath(s.root_mix, "test", "decomposition"), joinpath(s.root_mix, "test", "oracle", "mel"))
    sdr_dft = fetch(fi)
    sdr_mel = fetch(ft)
    info("Oracle SDR DFT($(s.feature["frame_length"])) = $(sdr_dft) dB")
    info("Oracle SDR Mel($(s.feature["mel_bands"])) = $(sdr_mel) dB")


    fi = remotecall(Mix.tensor, pid[1], s)
    ft = remotecall(Mix.tensor, pid[2], s, flag="test")
    fetch(fi)
    fetch(ft)
    info("tensors created")

    
    rmprocs(pid)
    info("worker process released")
    nothing
end
