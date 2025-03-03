using Plots, Plots.PlotMeasures, Distributions, Colors, Format

# ASEP_Z(N, t, q) returns the ASEP state vector (color of particle at each position) "state" after continuous time t
# and a negative number the_start which is the coordinate corresponding to the first entry of state, state[1].
#
# N is the initial size of the state; particles of color 1 to N are initially placed in reverse order and their positions
# are tracked. Particles of color N+1 enter from the left and particles of color 0 enter from the right; color merging allows
# us to assme that the colors of incoming particles are 0 and N+1 and ignore their true color. The state vector returned
# is actually the state at the random time \tau_t which is the last time before t at which a particle of color in [1,N]
# attempted a swap. If t is not too large for a given N, this will be off from the state at time t by O(1) many swaps where
# O(1) is a random quantity with exponential tails.

function ASEP_Z(N, t, q)
	state = reverse(collect(1:N));
	the_start=0;
	prob = q/(1+q);
	Infty=N+1;

	num_swaps = rand(Poisson(N*t*(1+q))); # number of swaps that the N particles we are tracking will attempt by time t
	k=1;

	while k <= num_swaps
		M = length(state)
		pos = rand(0:M+1) # picks a random position from the current positions in the state as well as the two boundary locations
		L = rand(Bernoulli(prob)); # probability that the attempted jump is to the left
		R = Bool(1-L);

		# We increment the swap count k only if the particle picked is one of the original N ones with color in [1,N].
		# The swap attempt still happens even otherwise, of course.
		if (0 < pos < M+1 && 0 < state[pos] < Infty)
			k += 1;
		end

		if (R && 0 < pos < M) # right swap attempt from bulk to bulk
			if (state[pos] > state[pos+1])
				state[pos], state[pos+1] = state[pos+1], state[pos];
			end
		elseif (L && 1 < pos < M+1) # left swap attempt from bulk to bulk or boundary
			if (state[pos] > state[pos-1])
				state[pos], state[pos-1] = state[pos-1], state[pos];
			end
		elseif (R && pos==0) # right swap attempt from boundary
			pushfirst!(state, state[1]);
			state[2] = Infty;
			the_start = the_start-1;
		elseif (R && pos == M) # right swap attempt to boundary
			push!(state, state[M]);
			state[M] = 0;
		end
	end

	return state, the_start
end


## ASEP_sheet calls ASEP_state to obtain the state at continuous time t and computes the height function of
## the state in a neighborhood of alpha in (-1,1) in the rarefaction fan. It then affinely shifts, recenters, and 
## scales the height function and plots it using the formulas in Section 2 (\epsilon is calculated from t) of 
## http://arxiv.org/abs/2403.01341. the_range parametrizes the domain on which the rescaled height_func is plotted,
## e.g., the_range = 3 means the plot will be on [-3,3].


function ASEP_sheet(alpha, q, the_range, t)
	mu = (1-alpha)^2/4;
	sigma_inv = 2(1-alpha^2)^(-2/3);
	beta = 2(1-alpha^2)^(1/3);
	gamma = 1-q;

	mu_prime = -(1-alpha)/2;

	eps = 2/(gamma*t);
	eps_inv = eps^(-1);
	eps_third = eps^(1/3);

	N = 5*Int(floor(abs(2*alpha*eps_inv) + beta*the_range*eps^(-2/3))); # 5 is an arbitrary choice so that the error in ASEP_Z will be small
	state, the_start = ASEP_Z(N, t, q);

	println("ASEP simulation complete.")

	cropsize = Int(floor(beta*eps^(-2/3)*the_range));
	height_func = fill(0, 2*cropsize+1, 2*cropsize+1);
	M = length(state);
	N_orig = Int(floor(N/2));

	for i in 1:(2*cropsize+1)
		for j in 1:(2*cropsize+1)
			height_func[i,j] = count(k->(k>=N_orig+cropsize-i), state[(N_orig -the_start + Int(floor(2*alpha*eps_inv)) - cropsize + j):M]);
		end
	end

	println("Height function calculation complete.")

	ASEP_sheet = fill(0.0, 2*cropsize+1, 2*cropsize+1);
	xrange = yrange = range(-the_range, the_range,2*cropsize+1);

	for i in -cropsize:cropsize
		for j in -cropsize:cropsize
			ASEP_sheet[cropsize+1+i, cropsize+1+j] = sigma_inv*eps_third*(mu*2*eps_inv + mu_prime*(j-i) - height_func[cropsize+1+i, cropsize+1 + j]); # + (xrange[cropsize+1+i]-yrange[cropsize+1+j])^2;
		end
	end

	# Plots command. To use the PlotlyJS backend of Plots to obtain interactive plots, call plotlyjs(); right before this command. You may have to use Pkg.add("PlotlyJS") the first time but I don't remember.
	plt = surface(xrange, yrange, ASEP_sheet, camera=(120,20), seriescolor=cgrad(:heat), dpi=300, legend = false, grid=false, size=(1000,1000), tickfontsize=14, background_color_outside=:white);

	return plt
end


# the function called for simulation of the ASEP_landscape, which needs multitime information on the state. 
# num_eval is number of times the state should be returned (they will be equally spaced), and t_start and t_end are
# the continuous time start and end for which ASEP is run

function ASEP_Z_multitime(N, q, t_start, t_end, num_eval) 
	state = reverse(collect(1:N));
	the_start= fill(0, num_eval);
	prob = q/(1+q);
	Infty=N+1;

	if (num_eval > 1)
		t_inc = (t_end - t_start)/(num_eval-1);
	else
		t_inc = 0;
	end

	num_swaps = fill(0, num_eval)
	num_swaps[1] = rand(Poisson(N*t_start*(1+q)));

	for i in 2:num_eval
		 num_swaps[i] = rand(Poisson(N*t_inc*(1+q)));
	end

	state_output = [Vector{Int}(undef,N) for _ in 1:num_eval]; # what we will output, it is a vector of vectors (initially empty) since we don't know the total number of swap attempts that will happen

	for i in 1:num_eval
		k=1;

		if (i > 1)
			the_start[i] = the_start[i-1];
		end

		while k <= num_swaps[i]
			M = length(state)
			pos = rand(0:M+1)
			L = rand(Bernoulli(prob));
			R = Bool(1-L);

			if (0 < pos < M+1 && 0 < state[pos] < Infty)
				k += 1;
			end

			if (R && 0 < pos < M)
				if (state[pos] > state[pos+1])
					state[pos], state[pos+1] = state[pos+1], state[pos];
				end
			elseif (L && 1 < pos < M+1)
				if (state[pos] > state[pos-1])
					state[pos], state[pos-1] = state[pos-1], state[pos];
				end
			elseif (R && pos==0)
				pushfirst!(state, state[1]);
				state[2] = Infty;
				the_start[i] = the_start[i]-1;
			elseif (R && pos==M)
				push!(state, state[M]);
				state[M] = 0;
			end

		end
		state_output[i] = copy(state);
	end

	return state_output, the_start
end



# Saves frames of ASEP landscape/directed landscape on [-space_range, space_range]^2, at macroscopic (i.e., 
# directed landscape) time 1 up to time t_range with num_frames many frames. T_start is the microscopic time (i.e., 
# time for the underlying ASEP) that the simulation is run for macroscopic time 1 and determines \eps in the
# definition of the ASEP landscape. You can put them together to make a video using a program like Blender.

function ASEP_landscape(alpha, q, space_range, T_start, t_range, num_frames)
	mu = (1-alpha)^2/4;
	sigma_inv = 2(1-alpha^2)^(-2/3);
	beta = 2(1-alpha^2)^(1/3);
	gamma = 1-q;

	mu_prime = -(1-alpha)/2;

	eps = 2/(gamma*T_start);
	eps_inv = eps^(-1);
	eps_third = eps^(1/3);

	T_end = T_start*t_range;

	N = 5*Int(floor(2*alpha*t_range*eps_inv + beta*space_range*eps^(-2/3)));

	state, the_start = ASEP_Z_multitime(N, q, T_start, T_end, num_frames);


	println("ASEP simulation complete.")

	cropsize = Int(floor(beta*eps^(-2/3)*space_range));


	z_lower_range = -4*space_range^2-1;


	for r in 1:num_frames
		height_func = fill(0, 2*cropsize+1, 2*cropsize+1);
		M = length(state[r]);
		K = Int(floor(N/2));

		if (num_frames > 1)
			t = 1.0 + (r-1)*(t_range-1)/(num_frames-1);
		else
			t = 1.0;
		end

		for i in 1:(2*cropsize+1)
			for j in 1:(2*cropsize+1)
				height_func[i,j] = count(k->(k>=K+cropsize-i), state[r][(K -the_start[r] + Int(floor(2*alpha*t*eps_inv)) - cropsize + j):M]);
			end
		end

		println("Height function calculation complete ("*string(r) * "/" *string(num_frames) * ").")

		ASEP_landscape = fill(0.0, 2*cropsize+1, 2*cropsize+1);
		xrange = yrange = range(-space_range, space_range,2*cropsize+1);

		for i in -cropsize:cropsize
			for j in -cropsize:cropsize
				ASEP_landscape[cropsize+1+i, cropsize+1+j] = sigma_inv*eps_third*(mu*t*2*eps_inv + mu_prime*(j-i) - height_func[cropsize+1+i, cropsize+1 + j]); # + (xrange[cropsize+1+i]-yrange[cropsize+1+j])^2;
			end
		end


		the_title = "\n\n\n                                       Time: "*format(t, precision=2);

		# Plots command
		plt = surface(xrange, yrange, ASEP_landscape, camera=(120,20), seriescolor=cgrad(:heat), dpi=300, legend = false, grid=false, size=(1000,1000), tickfontsize=14, background_color_outside=:white, zlims=(z_lower_range, 0), clims=(-z_lower_range, 0), title=the_title, topmargin=-150px, titlelocation=:center);


		
		savefig(plt, "ASEP_landscape_start=1_end="*string(t_range)*"_"*string(r)*".png");
	end
end
