#this file is part of litwin-kumar_et_al_dimension_2017
#Copyright (C) 2017 Ashok Litwin-Kumar
#see README for more information

using Distributions,LinearAlgebra,SpecialFunctions

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

#approximates the covariance matrix for gaussian patterns.  assumes each neuron's threshold is set to ensure identical sparsity
function cdist_gaus(J,f,Ndiscretization=4000)
	normcols = false
	normwsq = false

	h = range(-5,stop=5,length=Ndiscretization)
	dh = h[2]-h[1]
	hdist = Normal(0,1)
	Dh = dh*pdf.(hdist,h)


	sigsq = sum(J.*J,dims=2)
	sig = sigsq.^(1/2)
	sig12 = sigsq.^(1/4)
	rho = J*J'
	
	M = size(J,1)
	C = zeros(M,M)
	for ci = 1:M
		thetai = sqrt(2*sigsq[ci])*erfcinv(2*f)
		for cj = (ci+1):M
			if rho[ci,cj] == 0
				C[ci,cj] = 0
			elseif rho[ci,cj] >= (0.999*sig[ci]*sig[cj]) #perfectly correlated
				C[ci,cj] = f*(1-f)
			else
				thetaj = sqrt(2*sigsq[cj])*erfcinv(2*f)
				etai = sqrt(abs(rho[ci,cj])) * (sig12[ci]/sig12[cj])
				etaj = sign(rho[ci,cj])*sqrt(abs(rho[ci,cj])) * (sig12[cj]/sig12[ci])

				term2 = 0.25 * erfc.((thetai .- etai*h) / sqrt(2*(sigsq[ci] - etai^2)))  .*  erfc.((thetaj .- etaj*h) / sqrt(2*(sigsq[cj] - etaj^2)))

				C[ci,cj] = sum(Dh.*term2) - f^2
			end
		end
	end

	return C
end

#calculates dimension given the relevant statistics
function calcdim(M,f,mcij,varcij,varcii=0)
	if varcii == 0
		return (M*f^2 * (1-f)^2) ./ (f^2*(1-f)^2 + (M-1) * (mcij.^2 + varcij))
	else
		return (M*f^2 * (1-f)^2) ./ (f^2*(1-f)^2 + varcii + (M-1) * (mcij.^2 + varcij))
	end
end

M = 500
N = 100
K = 10
f = 0.1
syntype = "cortical" #"binary", "cortical", or "cerebellar"
J = genweights(M,N,K,syntype)
C = cdist_gaus(J,f)

C_upper = C[triu(ones(Bool,size(C)),1)]
mcij = mean(C_upper)
varcij = var(C_upper)

d = calcdim(M,f,mcij,varcij)
