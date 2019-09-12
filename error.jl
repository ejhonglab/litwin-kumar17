#this file is part of litwin-kumar_et_al_dimension_2017
#Copyright (C) 2017 Ashok Litwin-Kumar
#see README for more information

using Distributions 

#generate synaptic weight matrix with specified parameters
function genweights(M,N,K,syntype)
	J = zeros(M,N)

	for mi = 1:M
		inds = sample(1:N,K,replace=false)
		if syntype == "binary"
			J[mi,inds] = 1
		elseif (syntype == "cortical")
			mu = -.702
			sig = sqrt(.8752)
			J[mi,inds] = exp.(mu .+ sig*randn(K))
		elseif (syntype == "cerebellar")
			mu = 0
			sig = .438
			J[mi,inds] = exp.(mu .+ sig*randn(K))
		else
			error("invalid synapse type")
		end 
	end

	return J
end

#threshold the rows of h so each has coding level f
function threshold_sparsity(h,f)
	out = zeros(size(h))
	M = size(h,1)
	thresh = zeros(M)
	for mi = 1:M
		v = vec(h[mi,:])
		ha = sort(v)
		thresh[mi] = ha[round(Int,(1-f)*length(ha))]
		out[mi,:] = v .> thresh[mi]
	end
	return out,thresh
end

#threshold the rows of h using thresholds in thresh
function threshold(h,thresh)
	out = zeros(size(h))
	for mi = 1:M
		out[mi,:] = h[mi,:] .> thresh[mi]
	end
	return out
end


M = 500
N = 100
P = 100
K = 10
f = 0.1
eps = 0.3
syntype = "cortical" #"binary", "cortical", or "cerebellar"

J = genweights(M,N,K,syntype)

labels = rand(-1:2:1,P)
pats = randn(N,P) #training patterns
pats2 = pats .+ eps*randn(N,P) #noisy test patterns

h = J*pats
h2 = J*pats2

mpats,thresh = threshold_sparsity(h,f)
mpats2 = threshold(h2,thresh)


wm = (mpats .- f) * labels
h_classifier = ((mpats2' .- f)*wm)

error_rate = mean(sign.(h_classifier) .!=  labels)
