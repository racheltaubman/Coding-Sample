# Rachel Taubman Coding Sample 
# (Part of Senior Thesis in Physics at Scripps College)

import random
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import scipy.stats
from matplotlib.backends.backend_pdf import PdfPages
import itertools
from datetime import date

def nextstep(state, output):
    """takes in state and previous output (input for next step),
        and returns next state and ouput according to 
        hidden markov model
    """
    if state == "A":
        output = np.random.choice([0, 1])

        if output == 0:
            newstate = "A"
        elif output == 1:
            newstate = "B"

    elif state == "B":
        output = 1
        newstate = "A"

    return newstate, output


def Hopfield_with_Lrules(neurons, xs, J, h, T, n, H, eps, alpha, bounds):
    """ Simulates neurons as Hopfield network with learning rules
        for firing parameters J, T, and h--the learning rules 
        depend on parameters epsilon, alpha, H, and n. Returns the neurons'
        average energy usage (their average firing frequency)
        for the last 1,000 stimuli, the mean squared error (MSE) 
        of the neurons' prediction (their firing) compared to the stimuli
        for the last 1,000 stimuli, and the neurons' firing data for
        all stimuli. The MSE is calculated as one minus the coefficient
        of determination R^2 of the neurons' prediction.
    """
    firing_array = np.zeros([neurons, len(xs) - 1])
    # random initial firing values
    ws = [list(np.random.choice([0, 1], neurons)), 
        list(np.random.choice([0, 1], neurons))] 
    w = np.array(ws[1])
    # upper and lower bounds for J, T, and h values
    posbound = bounds * np.ones((neurons, neurons))
    negbound = -1 * bounds * np.ones((neurons, neurons))

    for t in range(1, len(xs)):
        # response to stimulus at timestep t
        w = np.dot(w, J) + xs[t] * h

        # firing occurs based on threshold T
        w[w > T] = 1
        w[w < T] = 0

        w = w.tolist()
        firing_array[:, t - 1] = w
        ws.append(w)
        ws_array = np.array(ws)

        # update J values according to J learning rule
        J = J + eps * (ws_array[t].reshape((neurons,1)) * ws_array[t - 1] \
                            - alpha * ws_array[t-1].reshape((neurons,1)) \
                            * ws_array[t - 1])
        J[J > posbound] = bounds
        J[J < negbound] = -bounds

        # update T values
        T = T + n * (ws_array[t] - H)
        T[T > posbound[0,:]] = bounds
        T[T < -posbound[0,:]] = -bounds

        # update h values
        h = h + eps * (ws_array[t] * xs[t - 1] \
                            - alpha * ws_array[t - 1] * xs[t - 1])
        h[h > posbound[0,:]] = bounds
        h[h < -posbound[0,:]] = -bounds

    reg = LinearRegression().fit(
        ws[-1001 : -1], xs[-1000 :]
        )

    return (
        np.mean(ws[-1001 : -1]),
        1 - reg.score(ws[-1001 : -1], xs[-1000 :]),
        firing_array
    )


def Hopfield_no_rules(neurons, xs, J, h):
    """ Simulates neurons as Hopfield network with no learning rules.
        Returns the neurons' average energy usage (their average firing 
        frequency) for the last 1,000 stimuli, the mean squared error (MSE) 
        of the neurons' prediction (their firing) compared to the stimuli
        for the last 1,000 stimuli, and the neurons' firing data for
        all stimuli. The MSE is calculated as one minus the coefficient
        of determination R^2 of the neurons' prediction.
    """
    firing_array = np.zeros([neurons, len(xs) - 1])
    w = np.random.choice([0, 1], neurons)
    ws = [list(w.tolist())]

    for t in range(1, len(xs)):
        w = np.dot(w, J) + xs[t] * h
        w[w < 1] = 0
        w[w >= 1] = 1

        w = w.tolist()
        ws.append(w)
        firing_array[:, t - 1] = w

    reg = LinearRegression().fit(
        ws[-1001 : -1], xs[-1000 :]
        )

    return (
        np.mean(ws[-1001 : -1]),
        1 - reg.score(ws[-1001 : -1], xs[-1000 :]),
        firing_array
    )


#####################################################################################
# Set up simulation info

# Info for file name
Filerun = 1
D = date.today()
today = D.strftime("%m%d%Y")

# define main parameters for the simulation run
# number of neurons
neurons = 3
# upper and lower bounds for parameters
bounds = 100
# number of timesteps for each run
trials = 2000
runs_of_trials = 3

# array of values for the parameters epsilon, alpha, and H
EAHarray = np.array([0, 1 / 3, 1 / 2, 2 / 3, 1])

# useful variables
LEAH = len(EAHarray)
subtrials = list(range(runs_of_trials))


# generate stimulus values, starting with random initial state
state = np.random.choice(["A", "B"])
state_list = [state]

if state == "A":
    output = np.random.choice([0, 1])
elif state == "B":
    output = 1

# list of ouputs from the Hidden Markov
# Process occuring for each timestep
xs = [output]

# Generate the outputs to show the
# neurons through nextstep function
for t in range(trials):
    state, output = nextstep(state, output)
    xs.append(output)
    state_list.append(state)

# take off first value so each output is
# at same timestep as its causal state
state_array = np.array(state_list)[1:]
xs_array = np.array(xs)[1:]
# indices that correspond to states A and B
A_indices = np.where(state_array == "A")
B_indices = np.where(state_array == "B")

# Create tuples (epsilon, alpha, H) for
# each combination of possible parameter values
EAHvals_iterated = list(itertools.product(
                        EAHarray.tolist(), repeat=3)
                        )
EAHvals = []
EAHvals_rounded = []
for i in range(len(EAHvals_iterated)):
    x = EAHvals_iterated[i]
    y = (round(x[0],2), round(x[1],2), round(x[2],2))
    EAHvals.append(x)
    EAHvals_rounded.append(y)

# Save tuples as strings for DataFrame indexing later
EAHvals_str = [str(EAHvals_rounded[i]) for i in range(len(EAHvals))]

# create arrays to store neuron firing data from 
# running the simulation with and without learning rules
noLR_firing = np.zeros([neurons, LEAH,
                                    LEAH, LEAH, 
                                    runs_of_trials, trials])

LR_firing = np.zeros([neurons, LEAH, 
                            LEAH, LEAH, 
                            runs_of_trials, trials])

# store change in MSE and change in energy
# in both data frame and numpy array
delta_MSE_energy_df = pd.DataFrame(
                        columns=subtrials, 
                        index=pd.MultiIndex.from_tuples(EAHvals)
                            )

# first row = change in MSE, second row = change in Energy
delta_MSE_energy_array = np.zeros([2, LEAH, LEAH, 
                                    LEAH, len(subtrials)])

# axis 0 = epsilon, axis 1 = alpha, axis 3 = H
a = 0
count = 0
for eps in EAHarray:
    b = 0
    for alpha in EAHarray:
        c = 0
        for H in EAHarray:
            for T in subtrials:
                # generate initial parameter values
                h0 = scipy.stats.norm.rvs(size=neurons)
                T0 = scipy.stats.norm.rvs(size=neurons)
                J0 = scipy.stats.norm.rvs(size=(neurons, neurons))
                n = 0.001

                # run simulation WITH the learning rules
                foo1 = Hopfield_with_Lrules(neurons, xs, J0, 
                                            h0, T0, n, H, 
                                            eps, alpha, bounds)
                # save average energy & mean squared error
                # for the last 1,000 timesteps 
                # and the firing data for all timesteps
                avg_energy_LR = foo1[0]
                MSE_LR = foo1[1]
                LR_firing[:, a, b, c, T, :] = foo1[2]

                # run simulation WITHOUT the learning rules
                foo2 = Hopfield_no_rules(neurons, xs, J0, h0)

                avg_energy_noLR = foo2[0]
                MSE_noLR = foo2[1]
                noLR_firing[:, a, b, c, T, :] = foo2[2]

                delta_MSE = MSE_LR - MSE_noLR
                delta_energy = avg_energy_LR - avg_energy_noLR

                delta_MSE_energy_df[T].loc[(eps, alpha, H)] = (
                    delta_MSE,
                    delta_energy
                    )

                delta_MSE_energy_array[0, a, b, c, T] = delta_MSE
                delta_MSE_energy_array[1, a, b, c, T] = delta_energy

            count += 1
            print('Completed parameter number {0} of {1}'.format(
                count,len(EAHvals))
                )
            c += 1
        b += 1
    a += 1


noLR_total_firing = np.zeros([neurons, LEAH, LEAH, LEAH])
LR_total_firing = np.zeros([neurons, LEAH, LEAH, LEAH])

# col 1 = state A, col 2 = state B
noLR_state_response = np.zeros([2, neurons, LEAH, LEAH, LEAH])
LR_state_response = np.zeros([2, neurons, LEAH, LEAH, LEAH])

# calculate the firing frequency of each neuron with
# each set of parameters in response to state A or B
# with and without learning rules
for n in range(neurons):
    for a in range(LEAH):
        for b in range(LEAH):
            for c in range(LEAH):
                noLR_total_firing = np.sum(
                    noLR_firing[n, a, b, c, :, :]
                )
                LR_total_firing = np.sum(
                    LR_firing[n, a, b, c, :, :]
                )

                if noLR_total_firing > 0:
                    noLR_state_response[0, n, a, b, c] = (
                        np.sum(
                            noLR_firing[
                                n, a, b, c, :, list(A_indices)
                            ]
                        )
                        / noLR_total_firing
                    )
                    noLR_state_response[1, n, a, b, c] = (
                        np.sum(
                            noLR_firing[
                                n, a, b, c, :, list(B_indices)
                            ]
                        )
                        / noLR_total_firing
                    )

                else:
                    noLR_state_response[0, n, a, b, c] = 0
                    noLR_state_response[1, n, a, b, c] = 0

                if LR_total_firing > 0:
                    LR_state_response[0, n, a, b, c] = (
                        np.sum(
                            LR_firing[
                                n, a, b, c, :, list(A_indices)
                            ]
                        )
                        / LR_total_firing
                    )
                    LR_state_response[1, n, a, b, c] = (
                        np.sum(
                            LR_firing[
                                n, a, b, c, :, list(B_indices)
                            ]
                        )
                        / LR_total_firing
                    )
                
                else:
                    LR_state_response[0, n, a, b, c] = 0
                    LR_state_response[1, n, a, b, c] = 0

# create column names and data array for 
# firing frequency data frame
firing_freq_columns = [
    "Prob(firing|state A) no LR",
    "Prob(firing|state A) LR",
    "Prob(firing|state B) no LR",
    "Prob (firing|state B) LR",
]

firing_freq_data = [
    noLR_state_response[0, :, :, :, :],
    noLR_state_response[1, :, :, :, :],
    LR_state_response[0, :, :, :, :],
    LR_state_response[1, :, :, :, :],
]

# create data frame with each entry in the form:
# (firing freq (FF) neuron 1, FF neuron 2, FF neuron 3)
firing_freq_df= pd.DataFrame(
    columns=firing_freq_columns, 
    index=pd.MultiIndex.from_tuples(EAHvals)
)

# enter firing frequency for each set of parameters in data frame
for col_name in firing_freq_columns:
    for EAHpoint in EAHvals:
        for data in firing_freq_data:
            EAH_list = EAHarray.tolist()
            i_eps = EAH_list.index(EAHpoint[0])
            i_alpha = EAH_list.index(EAHpoint[1])
            i_H = EAH_list.index(EAHpoint[2])
            data_point = []
            for n in range(neurons):
                data_point.append(data[n, i_eps, i_alpha, i_H])

            data_point_rounded = ["%.4f" % elem for elem in data_point]
            data_tuple = tuple(data_point_rounded)
            firing_freq_df[col_name].loc[
                (EAHpoint[0], EAHpoint[1], EAHpoint[2])
            ] = data_tuple

# save data
firing_freq_df.to_csv(
    r"C:\Users\taubm\Desktop\neuronproject" + 
    r"\firing_freq_df_{0}x{1}k_{2}_Run{3}.csv".format(
        runs_of_trials, trials, today, Filerun
        ), 
        index=False
)
                    
firing_freq_df.to_html(
    "firingfreqtable_{0}x{1}k_{2}_Run{3}.html".format(
        runs_of_trials, trials, today, Filerun
    )
)

np.save(
    "delta_MSE_energy_array_{0}x{1}k_{2}_Run{3}".format(
        runs_of_trials, trials, today, Filerun
    ),
    delta_MSE_energy_array
    )

np.save("state_array_{0}trials_{1}_Run{2}".format(
    trials, today, Filerun), 
    state_array
    )

delta_MSE_energy_df.to_csv(
    r"C:\Users\taubm\Desktop\neuronproject" +
    r"\DMSEEnergydf_{0}x{1}k_{2}_Run{3}.csv".format(
        runs_of_trials, trials, today, Filerun
        ),
        index=False
)

labels = EAHvals_str
# x = probability that MSE increase
# y = probability that energy decreases
X = [
    sum([delta_MSE_energy_df[:].loc[col][i][0] < 0 
    for i in subtrials]) /  
    len(subtrials)
    for col in EAHvals
]

Y = [
    sum([delta_MSE_energy_df[:].loc[col][i][1] > 0 
    for i in subtrials]) / 
    len(subtrials)
    for col in EAHvals
]

# create data frame with probabilities
# of MSE decreasing and energy increasing
delta_MSE_energy_prob_df = pd.DataFrame(
    columns=["Prob MSE decrease", 
             "Prob Energy increase"],
    index=pd.MultiIndex.from_tuples(EAHvals),
)

delta_MSE_energy_prob_df["Prob MSE decrease"] = X
delta_MSE_energy_prob_df["Prob Energy increase"] = Y

#sort data frame by MSE and energy probabilities then save as html
sorted_delta_MSE_energy_prob_df = delta_MSE_energy_prob_df.sort_values(
    by=["Prob MSE decrease", "Prob Energy increase"], 
    ascending=[True, False]
)

sorted_delta_MSE_energy_prob_df.to_html(
    "sorted_delta_MSE_energy_probs_"\
        "{0}x{1}k_{2}_Run{3}.html".format(
            runs_of_trials, trials, today, Filerun)
            )


#create PDF to save probabilities plot
pdf = PdfPages(
    "delta_MSE_energy_plot_{0}x{1}k_{2}_Run{3}.pdf".format(
        runs_of_trials, trials, today, Filerun
    )
)

fig, ax = plt.subplots(1, 1, figsize=(16, 12), sharex=True)

for x, y, lab in zip(X, Y, labels):
    ax.scatter(x, y, label=lab, s=80, alpha=0.8)

# set colormap w/ colors evenly spread out w/ number of datapoints
colormap = plt.cm.gist_ncar
colorst = [colormap(i) for i in np.linspace(0, 0.9, len(ax.collections))]
for t, j1 in enumerate(ax.collections):
    j1.set_color(colorst[t])

# create legend and label axes and title
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.6, box.height])
plt.rcParams.update({"font.size": 15})
plt.rc("figure", titlesize=18)
plt.rc('legend', fontsize=10)
plt.legend(bbox_to_anchor=(1,0.5), \
            loc = 'center left', \
            title = 'Parameters ('+u'\u03B5'+', '+u'\u03B1' + ', H)',\
            ncol = 3, borderaxespad = 2.0)
plt.xlabel("Probability of MSE decreasing", fontsize="large")
plt.ylabel("Probability of Energy increasing", fontsize="large")
plt.title("Probability of MSE decreasing vs" \
            "Probability of Energy\n increasing for" \
            "{0} neurons with changing parameters\n" \
            "(".format(neurons) 
            + u"\u03B5" 
            + ", " 
            + u"\u03B1" 
            + ", H) for {0} runs of {1} trials each".format(
                runs_of_trials, trials)
        )

pdf.savefig(fig)
plt.close()
pdf.close()

