using DataFrames
include("pdl.jl")



# bts = "balanced_train_segments.csv"
# es = "eval_segments.csv"
# uts = "unbalanced_train_segments.csv"


# for each worker, try 3 times before abandon
# write successful download to csv_done.csv
# write failures to csv_failure.csv
function dl_gas(csv, np)
    
    a = readdlm(csv)
    c = [(a[i,1],a[i,2],a[i,3],a[i,4]) for i = 1:size(a,1)]
    info("csv file loaded")

	# 20371×4 Array{Any,2}:
 	# "--4gqARaEJE,"  "0.000,"    "10.000,"   "/m/068hy,/m/07q6cd_,/m/0bt9lr,/m/0jbk"                               
 	# "--BfvyPmVMo,"  "20.000,"   "30.000,"   "/m/03l9g"                                                            
 	# "--U7joUcTCo,"  "0.000,"    "10.000,"   "/m/01b_21"                                                           

    addprocs(np)
    for i in workers()
        remotecall_fetch(include, i, "pdl.jl")
    end
    info("parallel session loaded")
    pmap(pdl, c)
    rmprocs(workers())
    nothing
end
