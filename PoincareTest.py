import sys
import random
import math
import os

import numpy as np
import copy

from datetime import datetime, timedelta

import mpi4py
mpi4py.rc.recv_mprobe = False

# remainder of program as before
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

initialSeed = 0

#random.seed(initialSeed)

tauSettings = [-2,-1.8,-1.6,-1.4,-1.2,-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1,1.2,1.4,1.6,1.8,2]
tauDeviations = [0, 0.25, 0.5, 0.75, 1]

tauSimSettings = []

for x in tauSettings:
	for y in tauDeviations:
		tauSimSettings.append([x,y])

#Input from header file
tauCenter = tauSettings[rank]
tauDeviation = 0

#Nodes / Edges
N = 100
E = 200

initialState = 0

#Outcome Matrix Setup
benefit = 10
cost = 8

R = benefit-cost
S = -cost
T = benefit
P = 0

outcomes = [[R , S],[T , P]]

#Network Details
network = []
neighbours = []
state = []
payoff = []
prosperity = []

#Decision Threshold Mode
# 1 - All nodes share the same threshold
# 2 - Nodes have their own threshold, to emulate diversity of opinion for private information decisions
tauMode = 2
#Decision Threshold
tau = 0

nodeThreshold = np.random.normal(loc=tauCenter, scale=tauDeviation, size=(N,))

#Weak - 0.001 Strong - 0.1
selectionStrength = 0.1
mutationThreshold = 0.0001

#InformationMode
# 1 - Only use private information
# 2 - Only use public information
# 3 - Use both types of information
infoMode = 3

#Public Info Mode
# 1 - Degree > Network Average
# 2 - Global History // Probablistic Aggregation
# 3 - Global History // Diminishing Probablistic Aggregation
# 4 - Global History // Bikhchandani

PublicInfoMode = 3

#Public Information History
# Format = NewcomerID, RolemodelID, Accepted Connections, Rejected Connections
globalHistory = []

#Format = Accepted Count, Rejected Count
globalHistoryCount = []

for i in range(N):
		globalHistoryCount.append([0,0])

globalHistoryLimit = 20

#Parameters for p (public info) and q (private info),
publicInformationThreshold = 0.75
privateInformationThreshold = 0.25

#Connection tracking
TruePositive = 0
FalsePositive = 0
TrueNegative = 0
FalseNegative = 0

#MinThreshold denotes when certain behaviour in main loop will begin to occur i.e. mutation, stat logging
roundMinThreshold = 10000
saveInterval = 10 ** 6
roundStatInterval = 10 ** 7
totalRounds = 10 ** 8
currentRound = 0

#Values for tracking transitions that occur during a simulation
transitionFlag = False
transitionStartFlag = False
numberOfTransitions = 0

#For logging mixed networks, Assume mixBalance = 50 will take a snapshot for 50 cooperators (update if more strategies are added to network)

haveLoggedMix = False
mixBalance = 50

neighbourMixSnap = []
stateMixSnap = []
tauMixSnap = []

#Local round stat tracking
averageCooperation = 0.0
averageDegree = 0.0
averageProsperity = 0.0
averagePayoff = 0.0

#Global stat tracking, data appended at round intervals defined by roundStatInterval
AverageCooperation = []
AverageDegree = []
AverageProsperity = []
AveragePayoff = []
AverageTau = []

#Summed values for stats, divided by totalRounds - roundMinThreshold yields mean average
coop = 0.0
degree = 0.0
prosp = 0.0
pay = 0.0

#Cascade tracking parameters
Ncascadeid = []
Pcascadeid = []
NodeNcascadeid = np.zeros(N)
NodeNcascadeLocalid = np.zeros(N)
NodePcascadeid = np.zeros(N)
NodePcascadeLocalid = np.zeros(N)
Ncascadesize = []
Pcascadesize = []

Ncascadelog = {}
Pcascadelog = {}
minCascadeSize = 4

#For addressing memory issues in longer term simulations
CurrentOffload = 0

#Collections for storing network snapshots at periodic points
neighbourSnap = []
stateSnap = []
tauSnap = []

#Snapshots for logging point of transitions
neighbourTranSnap = []
stateTranSnap = []
tauTranSnap = []

#Markov Chain Variables
#States - 0-C, 1-D, 2-MC, 3-MD

markovThreshold = 10 ** 9
stateHistory = []
stateChangeProbability = [1.0 - mutationThreshold, 0.0, 0.0, mutationThreshold]
stateChangeOccurences = {}
coopProbsSum = 0
runningCoopProbTotal = []
defectProbsSum = 0
runningDefectProbTotal = []
runningCoopProbTotalSummary = []


stateHistory = []
stateChangeProbability = [1.0 - mutationThreshold, 0.0, 0.0, mutationThreshold]
stateChangeOccurences = {}
coopProbsSum = 0
runningCoopProbTotal = []
defectProbsSum = 0
runningDefectProbTotal = []
runningCoopProbTotalSummary = []

tauDevSubDirectory = "SimResults/GH/" + str(PublicInfoMode)

if not os.path.exists(tauDevSubDirectory):
	os.mkdir(tauDevSubDirectory)

directoryLocation = tauDevSubDirectory + "/tauM_" + str(tauMode) +"_selcStr_" + str(selectionStrength) + "_mutRate_" + str(mutationThreshold) +"_tauCen_" + str(tauCenter) + "_q_" + str(privateInformationThreshold) + "_publicInfoMode_" + str(PublicInfoMode) + "_historyLimit_" + str(globalHistoryLimit)

if not os.path.exists(directoryLocation):    
	os.mkdir(directoryLocation)

cascadeDirectoryLocation = directoryLocation + "/Cascades"

if not os.path.exists(cascadeDirectoryLocation):    
	os.mkdir(cascadeDirectoryLocation)

def saveState():
    
	saveValues = {}

	global directoryLocation, initialSeed
	global network, neighbours, state, payoff, prosperity
	global nodeThreshold, globalHistory, globalHistoryCount, globalHistoryLimit, Ncascadelog, Pcascadelog
	global TruePositive, FalsePositive, TrueNegative, FalseNegative
	global currentRound, transitionFlag, transitionStartFlag, numberOfTransitions    
	global AverageCooperation, AverageDegree, AverageProsperity, AveragePayoff, AverageTau, coop, degree, prosp, pay
	global NodeNcascadeLocalid, NodePcascadeLocalid
	global Ncascadeid, Pcascadeid, NodeNcascadeid, NodePcascadeid, Ncascadesize, Pcascadesize
	global neighbourSnap, stateSnap, tauSnap, neighbourTranSnap, stateTranSnap, tauTranSnap     
	global haveLoggedMix, mixBalance, neighbourMixSnap, stateMixSnap, tauMixSnap

	saveValues["initialSeed"] = initialSeed
	saveValues["network"] = network.tolist()
	saveValues["neighbours"] = neighbours
	saveValues["state"] = state
	saveValues["payoff"] = payoff
	saveValues["prosperity"] = prosperity
	saveValues["nodeThreshold"] = nodeThreshold.tolist()
	saveValues["globalHistory"] = globalHistory
	saveValues["globalHistoryCount"] = globalHistoryCount
	saveValues["TruePositive"] = TruePositive
	saveValues["FalsePositive"] = FalsePositive
	saveValues["TrueNegative"] = TrueNegative
	saveValues["FalseNegative"] = FalseNegative
	saveValues["currentRound"] = currentRound
	saveValues["transitionFlag"] = transitionFlag
	saveValues["transitionStartFlag"] = transitionStartFlag
	saveValues["numberOfTransitions"] = numberOfTransitions
	saveValues["AverageCooperation"] = AverageCooperation
	saveValues["AverageDegree"] = AverageDegree
	saveValues["AverageProsperity"] = AverageProsperity
	saveValues["AveragePayoff"] = AveragePayoff
	saveValues["AverageTau"] = AverageTau
	saveValues["coop"] = coop
	saveValues["degree"] = degree
	saveValues["prosp"] = prosp
	saveValues["pay"] = pay
	saveValues["NodeNcascadeLocalid"] = NodeNcascadeLocalid.tolist()
	saveValues["NodePcascadeLocalid"] = NodePcascadeLocalid.tolist()
	saveValues["Ncascadeid"] = Ncascadeid
	saveValues["Pcascadeid"] = Pcascadeid
	saveValues["NodeNcascadeid"] = NodeNcascadeid.tolist()
	saveValues["NodePcascadeid"] = NodePcascadeid.tolist()
	saveValues["Ncascadesize"] = Ncascadesize
	saveValues["Pcascadesize"] = Pcascadesize
	saveValues["neighbourSnap"] = neighbourSnap
	saveValues["stateSnap"] = stateSnap
	saveValues["tauSnap"] = tauSnap
	saveValues["neighbourTranSnap"] = neighbourTranSnap
	saveValues["stateTranSnap"] = stateTranSnap
	saveValues["tauTranSnap"] = tauTranSnap

	saveValues["haveLoggedMix"] = haveLoggedMix
	saveValues["mixBalance"] = mixBalance
	saveValues["neighbourMixSnap"] = neighbourMixSnap
	saveValues["stateMixSnap"] = stateMixSnap
	saveValues["tauMixSnap"] = tauMixSnap

	saveValues["Ncascadelog"] = copy.deepcopy(Ncascadelog)
	saveValues["Pcascadelog"] = copy.deepcopy(Pcascadelog)

	for currVal in list(saveValues["Ncascadelog"].keys()):
		saveValues["Ncascadelog"][currVal][3] = 0

	for currVal in list(saveValues["Pcascadelog"].keys()):
		saveValues["Pcascadelog"][currVal][3] = 0

	with open(directoryLocation + "/saveState.txt", "w") as text_file:
		print(saveValues, file = text_file)
        


def loadState():
	global directoryLocation
    
	if(os.path.exists(directoryLocation + "/saveState.txt")):
		saveValues = {}
        
		global initialSeed, network, neighbours, state, payoff, prosperity
		global nodeThreshold, globalHistory, globalHistoryCount, globalHistoryLimit, Ncascadelog, Pcascadelog
		global TruePositive, FalsePositive, TrueNegative, FalseNegative
		global currentRound, transitionFlag, transitionStartFlag, numberOfTransitions    
		global AverageCooperation, AverageDegree, AverageProsperity, AveragePayoff, AverageTau, coop, degree, prosp, pay
		global NodeNcascadeLocalid, NodePcascadeLocalid
		global Ncascadeid, Pcascadeid, NodeNcascadeid, NodePcascadeid, Ncascadesize, Pcascadesize
		global neighbourSnap, stateSnap, tauSnap, neighbourTranSnap, stateTranSnap, tauTranSnap 
		global haveLoggedMix, mixBalance, neighbourMixSnap, stateMixSnap, tauMixSnap

		content = []
		with open(directoryLocation + "/saveState.txt") as text_file:
		    content = text_file.readlines()

		content = [x.strip() for x in content]

		finalContent = ""

		for curr in content:
			finalContent += curr.replace("array", "")
		
		saveValues = eval(finalContent)
	    
		initialSeed = saveValues["initialSeed"]
		network = np.array(saveValues["network"])
		neighbours = saveValues["neighbours"]
		state = saveValues["state"]
		payoff = saveValues["payoff"]
		prosperity = saveValues["prosperity"]
		nodeThreshold = np.array(saveValues["nodeThreshold"])
		globalHistory = saveValues["globalHistory"]
		globalHistoryCount = saveValues["globalHistoryCount"]
		TruePositive = saveValues["TruePositive"]
		FalsePositive = saveValues["FalsePositive"]
		TrueNegative = saveValues["TrueNegative"]
		FalseNegative = saveValues["FalseNegative"]
		currentRound = saveValues["currentRound"]
		transitionFlag = saveValues["transitionFlag"]
		transitionStartFlag = saveValues["transitionStartFlag"]
		numberOfTransitions = saveValues["numberOfTransitions"]
		AverageCooperation = saveValues["AverageCooperation"] 
		AverageDegree = saveValues["AverageDegree"]
		AverageProsperity = saveValues["AverageProsperity"]
		AveragePayoff = saveValues["AveragePayoff"]
		AverageTau = saveValues["AverageTau"]
		coop = saveValues["coop"]
		degree = saveValues["degree"]
		prosp = saveValues["prosp"]
		pay = saveValues["pay"]
		Ncascadelog = saveValues["Ncascadelog"]
		Pcascadelog = saveValues["Pcascadelog"]
		NodeNcascadeLocalid = np.array(saveValues["NodeNcascadeLocalid"])
		NodePcascadeLocalid = np.array(saveValues["NodePcascadeLocalid"])
		Ncascadeid = saveValues["Ncascadeid"]
		Pcascadeid = saveValues["Pcascadeid"]
		NodeNcascadeid = np.array(saveValues["NodeNcascadeid"])
		NodePcascadeid = np.array(saveValues["NodePcascadeid"])
		Ncascadesize = saveValues["Ncascadesize"]
		Pcascadesize = saveValues["Pcascadesize"]
		neighbourSnap = saveValues["neighbourSnap"]
		stateSnap = saveValues["stateSnap"]
		tauSnap = saveValues["tauSnap"]
		neighbourTranSnap = saveValues["neighbourTranSnap"]
		stateTranSnap = saveValues["stateTranSnap"]
		tauTranSnap = saveValues["tauTranSnap"]

		haveLoggedMix = saveValues["haveLoggedMix"]
		mixBalance = saveValues["mixBalance"]
		neighbourMixSnap = saveValues["neighbourMixSnap"]
		stateMixSnap = saveValues["stateMixSnap"]
		tauMixSnap = saveValues["tauMixSnap"]

		for currVal in list(Ncascadelog.keys()):
			Ncascadelog[currVal][3] = datetime.now() + timedelta(minutes = 30)

		for currVal in list(Pcascadelog.keys()):
			Pcascadelog[currVal][3] = datetime.now() + timedelta(minutes = 30)

	else:
		epoch = datetime.utcfromtimestamp(0)
		initialSeed = (int((datetime.now() - epoch).total_seconds() * 1000))

		random.seed(initialSeed)


def networkCreation(N, E):
	edges = []
		
	for i in range(0,N):
		for j in range(0,i):
			edges.append([i,j])
	
	np.random.shuffle(edges)
		
	network = np.zeros((N,N))
		
	for i in range(0,E):
		a = edges[i][0]
		b = edges[i][1]
	
		network[a][b] = 1
		network[b][a] = 1
		
	return network   
	

def prosperityTargeting(rand, prosperity):
		#Select node to target as rolemodel using prosperity/fitness

		node = 0
		proslen = len(prosperity)
		prossum = sum(prosperity)

		if(prossum == 0):
			node = -1
		else:
			culpros = [0.0 for i in range(proslen)]

			for i in range(proslen):
				if i==0:
					culpros[i] = 1.0 * prosperity[i] / prossum
				else:
					culpros[i] = culpros[i - 1] + 1.0 * prosperity[i] / prossum

				left = 0
				right = proslen - 1

				if(rand <= culpros[0]):
					node = 0
				else:
					while ((right - left) > 1):
						middle = int(math.floor((left + right) / 2))
						if(rand <= culpros[middle]):
							right = middle
						else:
							left = middle
						node = right

		return node


network = networkCreation(N, E)
neighbours = [[] for i in range(N)]

#Initalises inital behaviour state of all nodes in network
state = [initialState] * N

#Initialises inital payoff of all nodes
payoff = [0.0] * N

prosperity = [0.0] * N

#Populate neighbours array
for i in range(N):
	for j in range(N):
		if(network[i][j] == 1):
			neighbours[i].append(j)

#Calculate initial payoff and prosperity fitness for each node in the inital network
for i in range(N):
	for j in neighbours[i]:
		payoff[i] = payoff[i] + outcomes[state[i]][state[j]]
	prosperity[i] =  pow(1+selectionStrength, payoff[i])   
	
averageCooperation = 0.0
averageDegree = 0.0
averageProsperity = 0.0
averagePayoff = 0.0
averageTau = 0.0

coop = 0.0
degree = 0.0
prosp = 0.0
pay = 0.0

loadState()

print("\n\n--------------------")
print("Beginning Simulation...")
print(datetime.now())
print("Simulation Parameters")
print("--------------------")
print("Tau Settings - Centre : " + str(tauCenter) + ", Deviation : " + str(tauDeviation))
print("Selection Strength : " + str(selectionStrength))
print("Mutation Rate : " + str(mutationThreshold))
print("Information Mode : " + str(infoMode))
print("Starting Round : " + str(currentRound))

while currentRound < totalRounds:

	#New node index
	newNode = np.random.randint(0, N)
	newNodeTau = np.random.normal(loc=tauCenter, scale=tauDeviation)

	#Determine role model for new node
	rand = np.random.rand(1)    
	roleModel = prosperityTargeting(rand, prosperity)

	didMutate = False

	#Determine if new node will adopt rolemodel behaviour or mutate   
	if(currentRound <= roundMinThreshold):
		#If in early stage, just adopt rolemodel behaviour
		tempStateNewNode = state[roleModel]
	else:
		#Check if newly added node mutates to behaviour that opposes rolemodel
		rand = np.random.rand(1)
		if(rand < mutationThreshold):
			tempStateNewNode = 1 - state[roleModel]
			didMutate = True
		else:
			tempStateNewNode = state[roleModel]

	if(currentRound >= markovThreshold):
		stateHistory.append([stateChangeProbability.copy(), [tempStateNewNode, didMutate]])
		stateTag = str(stateChangeProbability[0]) + " - " + str(stateChangeProbability[1])

		coopProbsSum += stateChangeProbability[0]
		runningCoopProbTotal.append(coopProbsSum / len(stateHistory))

		defectProbsSum += 1 - stateChangeProbability[0]
		runningDefectProbTotal.append(defectProbsSum / len(stateHistory))

		if(stateTag in stateChangeOccurences):
			stateChangeOccurences[stateTag] += 1
		else:
			stateChangeOccurences[stateTag] = 1

	#Initalise array for new node neigbour and load neigbours of role model
	newNeigboursTemp = []
	roleNeighboursTemp = []

	roleNeighboursTemp.append(roleModel)
	for newNeighbour in neighbours[roleModel]:
		roleNeighboursTemp.append(newNeighbour)



	CurrentAccepted = []
	CurrentRejected = []

	#Check each new potential neighbour to determine if new node will form connection
	for newNeighbour in roleNeighboursTemp:

		PCascade = False
		NCascade = False

		nid = 0
		pid = 0

		if(newNeighbour not in newNeigboursTemp):

			#Calculate private information based on behaviour of new potential neighbour, see 'Decisions based on private information' 
			if(state[newNeighbour] == 0):
				s = np.random.normal(-0.5, np.sqrt(0.5), 1)
			else:
				s = np.random.normal(0.5, np.sqrt(0.5), 1)


			#Determine if private connection flag is raised
			privateConnect = False

			if(tauMode == 1):
				if(s[0] < tau):
					privateConnect = True
			elif(tauMode == 2):
				if(s[0] < newNodeTau):
					privateConnect = True

			#Determine if public connection flag is raised, will be true if degree of neigbour node exceeds average of network
			publicConnect = False

			overrideConnect = False
			overrideValue = False
            
			if(PublicInfoMode == 1):
				#Degree > Network Average Degree
				if(len(neighbours[newNeighbour]) > averageDegree):
					publicConnect = True
			elif(PublicInfoMode == 2):
				# Probabilistic Aggregation W/ Global History
				newNeigbourScore = globalHistoryCount[newNeighbour]
				newNeigbourScoreTotal = sum(newNeigbourScore)

				if(newNeigbourScoreTotal <= 1):
					#Coin Flip
					publicConnect = (random.uniform(0,1) > 0.5)
				else:
					#Check proportional opinion of new node
					publicConnect = (random.uniform(0,1) > 1 - (newNeigbourScore[0] / newNeigbourScoreTotal))
			elif(PublicInfoMode == 3):
				#Diminishing Probabilistic Aggregation W/ Global History
				
				infoMod = 1 / globalHistoryLimit
				plusTotal = 0
				minusTotal = 0
				
				for i in range(0,len(globalHistory)):
					currScores = globalHistory[i]
					currMod = infoMod * (globalHistoryLimit - (i + 1))
					
					if(newNeighbour in currScores[2]):
					    plusTotal += 1 - currMod
						
					if(newNeighbour in currScores[3]):
					    minusTotal += 1 - currMod
						
				newNeigbourScoreTotal = plusTotal + minusTotal
				
				if(newNeigbourScoreTotal <= 1):
				    #Coin Flip
					publicConnect = (random.uniform(0,1) > 0.5)
				else:
				    #Check proportional opinion of new node
					publicConnect = (random.uniform(0,1) > 1 - (plusTotal / newNeigbourScoreTotal))
					
			elif(PublicInfoMode ==4):
				#Bikhchandani Method W/ Global History
				newNeigbourScore = globalHistoryCount[newNeighbour]
				difference = newNeigbourScore[0] - newNeigbourScore[1]

				if(difference > 1):
					#Connect regardless of private signal
					publicConnect = True
                    
					overrideConnect = True
					overrideValue = True
                    
				if(difference == 1):
					#Adopt if p is high, else toss a coin 
					if(privateConnect):
						publicConnect = True
					else:
						publicConnect = (random.uniform(0,1) > 0.5)
				if(difference == -1):
					#Reject if p is low, else toss a coin
					if(privateConnect):
						publicConnect = (random.uniform(0,1) > 0.5)
				if(difference < -1):
					#Reject regardless of private signal
					publicConnect = False
                    
					overrideConnect = True
					overrideValue = False


			connect = False

			if(infoMode == 1):
				connect = privateConnect
			elif(infoMode == 2):
				connect = publicConnect
			else:
				#Decide if connection should be formed based on flags
				# pri && pub == connect
				# !pri && !pub == don't connect
				# pri && !pub == use p parameter
				# !pri && pub == use q parameter
				if(privateConnect and publicConnect):
					connect = True
				elif((not privateConnect) and (not publicConnect)):
					connect = False
				elif(publicConnect and (not privateConnect)):
					if(np.random.uniform(0,1) < publicInformationThreshold):
						connect=True
				elif((not publicConnect) and (privateConnect)):
					if(np.random.uniform(0,1) < privateInformationThreshold):
						connect=True

			if(overrideConnect):
				connect = overrideValue

			#Log results of decision (TP,FP,TN,FN)
			if connect :                    
				if (state[newNeighbour] == 0):
					TruePositive += 1
				else:
					FalsePositive += 1
					if(not privateConnect):
						NCascade = True
			else:
				if(state[newNeighbour] == 1):
					TrueNegative += 1
				else:
					FalseNegative += 1
					if(not publicConnect):
						PCascade = True

			#Add new neighbour to new nodes neighbours if connection was made
			if(newNode != newNeighbour):  

				if(connect):
					newNeigboursTemp.append(newNeighbour)
					CurrentAccepted.append(newNeighbour)
				else:
					CurrentRejected.append(newNeighbour)

	if(currentRound >= roundMinThreshold):
		#Log if cascade took place in previous step
		# N-Cascade - When public information indicates a connection should not be made, but is made regardless        
		if(NCascade):
			nid = int(NodeNcascadeid[roleModel])
			if nid == 0:
				nid = int(len(Ncascadeid) + 1)
				Ncascadeid.append(nid)
				Ncascadesize.append(0)                    

			NodeNcascadeid[newNode] = nid
			Ncascadesize[nid-1] += 1

			if str(nid) not in Ncascadelog:
				Ncascadelog[str(nid)] = [[0], [], [currentRound], datetime.now() + timedelta(minutes = 15)]
				NodeNcascadeLocalid[newNode] = 0
			else:
				logId = int(Ncascadelog[str(nid)][0][-1] + 1)
				Ncascadelog[str(nid)][0].append(logId)
				Ncascadelog[str(nid)][1].append([logId, int(NodeNcascadeLocalid[roleModel])])
				Ncascadelog[str(nid)][2].append(currentRound)
				Ncascadelog[str(nid)][3] = datetime.now() + timedelta(minutes = 15)
				NodeNcascadeLocalid[newNode] = logId

		else:
			NodeNcascadeid[newNode] = 0
			NodeNcascadeLocalid[newNode] = 0

		# P-Cacade - When private information indicates a connection should be made, but is not made
		if(PCascade):
			pid = int(NodePcascadeid[roleModel])
			if pid == 0:
				pid = int(len(Pcascadeid) + 1)
				Pcascadeid.append(pid)
				Pcascadesize.append(0)

			NodePcascadeid[newNode] = pid
			Pcascadesize[pid-1] += 1                

			if str(pid) not in Pcascadelog:
				Pcascadelog[str(pid)] = [[0], [], [currentRound], datetime.now() + timedelta(minutes = 15)]
				NodePcascadeLocalid[newNode] = 0
			else:
				logId = int(Pcascadelog[str(pid)][0][-1] + 1)                    
				Pcascadelog[str(pid)][0].append(logId)
				Pcascadelog[str(pid)][1].append([logId, int(NodePcascadeLocalid[roleModel])])
				Pcascadelog[str(pid)][2].append(currentRound)
				Pcascadelog[str(pid)][3] = datetime.now() + timedelta(minutes = 15)
				NodePcascadeLocalid[newNode] = logId

		else:
			NodePcascadeid[newNode] = 0
			NodePcascadeLocalid[newNode] = 0

	#Remove neighbours for node being replaced and adjust stats for all nodes that were connected
	oldNodeNeighbourTemp = copy.deepcopy(neighbours[newNode])
	for oldNeighbour in oldNodeNeighbourTemp:
		tempid = neighbours[oldNeighbour].index(newNode)
		del neighbours[oldNeighbour][tempid]
		payoff[oldNeighbour] = payoff[oldNeighbour] - outcomes[state[oldNeighbour]][state[newNode]]
		prosperity[oldNeighbour] = pow(1 + selectionStrength, payoff[oldNeighbour])
		tempid = neighbours[newNode].index(oldNeighbour)
		del neighbours[newNode][tempid]
		network[newNode][oldNeighbour] = 0
		network[oldNeighbour][newNode] = 0

	if(len(neighbours[newNode]) > 0):
		print("Failed to remove old neigbours")


	#Initalise stats for newly added node, including it's interactions with its new neigbours added in previous stepws
	state[newNode] = tempStateNewNode
	payoff[newNode] = 0.0
	prosperity[newNode] = 1.0
	nodeThreshold[newNode] = newNodeTau
	globalHistoryCount[newNode] = [0,0]


	for histEntry in globalHistory:
		if(newNode in histEntry[2]):
			histEntry[2].remove(newNode)
		if(newNode in histEntry[3]):
			histEntry[3].remove(newNode)

	for initNewConnection in newNeigboursTemp:
		neighbours[newNode].append(initNewConnection)
		neighbours[initNewConnection].append(newNode)
		network[newNode][initNewConnection] = 1
		network[initNewConnection][newNode] = 1        
		payoff[newNode] = payoff[newNode] + outcomes[state[newNode]][state[initNewConnection]]
		prosperity[newNode] = pow(1 + selectionStrength, payoff[newNode])
		payoff[initNewConnection] = payoff[initNewConnection] + outcomes[state[initNewConnection]][state[newNode]]
		prosperity[initNewConnection] = pow(1 + selectionStrength, payoff[initNewConnection])


	#Update Global History

	if(len(globalHistory) >= globalHistoryLimit):
		oldHistory = globalHistory[0]

		for accepted in oldHistory[2]:
			globalHistoryCount[accepted][0] -= 1

		for rejected in oldHistory[3]:
			globalHistoryCount[rejected][1] -= 1

		del globalHistory[0]

	for accepted in CurrentAccepted:
		globalHistoryCount[accepted][0] += 1

	for rejected in CurrentRejected:
		globalHistoryCount[rejected][1] += 1

	globalHistory.append([newNode, roleModel, CurrentAccepted.copy(), CurrentRejected.copy()])

	#Check if transition has started
	if(sum(state) != 0 and transitionStartFlag == False and transitionFlag == False):
		transitionStartFlag = True
		haveLoggedMix = False

	#If invader transition has completed, log it and mark transition as ended 
	if(sum(state) == N and transitionStartFlag == True and transitionFlag == False):
		transitionFlag = True
		numberOfTransitions += 1
		transitionStartFlag = False

		neighbourTranSnap.append(copy.deepcopy(neighbours))
		stateTranSnap.append(state.copy())
		tauTranSnap.append(nodeThreshold.copy())

	#If cooperation transition has completed, log it and marks transition as ended
	if(sum(state) == 0 and transitionStartFlag == False and transitionFlag == True):
		transitionFlag = False
		numberOfTransitions += 1
		transitionStartFlag = True

		neighbourTranSnap.append(copy.deepcopy(neighbours))
		stateTranSnap.append(state.copy())
		tauTranSnap.append(nodeThreshold.copy())

	cooperationProsperity = 0.0

	#Calculate averages
	for node in range(N):
		averageDegree = averageDegree + len(neighbours[node])
		averageProsperity = averageProsperity + prosperity[node]
		averagePayoff = averagePayoff + payoff[node]
		averageTau = averageTau + nodeThreshold[node]
		if(state[node] == 0):
			cooperationProsperity = cooperationProsperity + prosperity[node]


	averageCooperation = N - sum(state)
	averageDegree = averageDegree / N
	averageProsperity = averageProsperity / N
	averagePayoff = averagePayoff / N
	averageTau = averageTau / N

	if(currentRound >= roundMinThreshold):
		coop += averageCooperation
		degree += averageDegree
		prosp += averageProsperity
		pay += averagePayoff


	probabilityMutCoop = ((sum(prosperity) - cooperationProsperity) / sum(prosperity)) * mutationThreshold
	probabilityMutDefect  = (cooperationProsperity / sum(prosperity)) * mutationThreshold

	probabilityCoop = (cooperationProsperity / sum(prosperity))
	probabilityDefect = ((sum(prosperity) - cooperationProsperity) / sum(prosperity))

	if(probabilityCoop != 1):
		probabilityCoop = probabilityCoop - probabilityMutDefect
		probabilityDefect = probabilityDefect - probabilityMutCoop
	else:
		if(probabilityCoop == 1):
			probabilityCoop = probabilityCoop - probabilityMutDefect
		if(probabilityDefect == 1):
			probabilityDefect = probabilityDefect - probabilityMutCoop

	stateChangeProbability = [probabilityCoop, probabilityDefect, probabilityMutCoop, probabilityMutDefect]

	if((currentRound % roundStatInterval) == 0):
		print(str((currentRound / roundStatInterval) * 10) + "%")
		print(datetime.now())
		AverageCooperation.append(averageCooperation)
		AverageDegree.append(averageDegree)
		AverageProsperity.append(averageProsperity)
		AveragePayoff.append(averagePayoff)
		AverageTau.append(averageTau)
		neighbourSnap.append(copy.deepcopy(neighbours))
		stateSnap.append(state.copy())
		tauSnap.append(nodeThreshold.copy())

	if(transitionStartFlag and len(neighbourMixSnap) < 10 and not haveLoggedMix):
		if(sum(state) == mixBalance):
			neighbourMixSnap.append(copy.deepcopy(neighbours))
			stateMixSnap.append(state.copy())
			tauMixSnap.append(nodeThreshold.copy())
			haveLoggedMix = True
		
	if((currentRound % (roundStatInterval / 2)) == 0):
		#Offload cascade details into seperate file here to address memory issues
		#cascadeDirectoryLocation

		CurrentOffload += 1
		NCascadeOffload = {}
		PCascadeOffload = {}
		
		print("NCascade Count - " + str(len(Ncascadelog)))

		for currVal in list(Ncascadelog.keys()):
			curr = str(currVal)
			if(datetime.now() > Ncascadelog[curr][3]):
				NCascadeOffload[curr] = (Ncascadelog[curr])
				del Ncascadelog[curr]

		if(len(NCascadeOffload) > 0):
			with open(cascadeDirectoryLocation + "/NCascade_" + str(CurrentOffload) +"_Offload.txt", "w") as text_file:
				print(NCascadeOffload, file = text_file)

		for currVal in list(Pcascadelog.keys()):
			curr = str(currVal)
			if(datetime.now() > Pcascadelog[curr][3]):
				PCascadeOffload[curr] = (Pcascadelog[curr])
				del Pcascadelog[curr]

		if(len(PCascadeOffload) > 0):
			with open(cascadeDirectoryLocation + "/PCascade_" + str(CurrentOffload) +"_Offload.txt", "w") as text_file:
				print(PCascadeOffload, file = text_file)

			

		

	currentRound += 1

	if(((currentRound - 1) % saveInterval) == 0 and (currentRound - 1) > 0):
		saveState()

print("Simulation completed - " +  str(tauCenter) + " - " + str(tauDeviation))
print(datetime.now())
print("--------------------------------")

#Return all values to header file here
#Maybe just write them in Poincare?

with open(directoryLocation + "/snaps.txt", "w") as text_file:
	print(neighbourSnap, file = text_file)
	print(stateSnap, file = text_file)
	print(tauSnap, file = text_file)

with open(directoryLocation + "/tranSnaps.txt", "w") as text_file:
	print(neighbourTranSnap, file = text_file)
	print(stateTranSnap, file = text_file)
	print(tauTranSnap, file = text_file)

with open(directoryLocation + "/NcascadeTrace.txt", "w") as text_file:
	print(Ncascadelog, file = text_file)

with open(directoryLocation + "/PcascadeTrace.txt", "w") as text_file:
	print(Pcascadelog, file = text_file)

infoModeStr = ["Private Only", "Public Only", "Public & Private"]

with open(directoryLocation + "/results.txt", "w") as text_file:
        print("Simulation Parameters", file = text_file)
        print("--------------------", file = text_file)
        print("Tau Settings - Centre : " + str(tauCenter) + ", Deviation : " + str(tauDeviation), file = text_file)
        print("Selection Strength : " + str(selectionStrength), file = text_file)
        print("Mutation Rate : " + str(mutationThreshold), file = text_file)
        print("Information Mode : " + infoModeStr[infoMode - 1], file = text_file)

        if(infoMode == 2):
            print("Public Information Threshold : ", str(publicInformationThreshold), file = text_file)
            print("Private Information Threshold : " , str(privateInformationThreshold), file = text_file)

        print("\nOverall Stats", file = text_file)
        print("--------------------", file = text_file)

        print("Average Cooperation : " , (coop / (totalRounds - roundMinThreshold)), file = text_file)
        print("Average Degree : " , (degree / (totalRounds - roundMinThreshold)), file = text_file)
        print("Average Prosperity : " , (prosp / (totalRounds - roundMinThreshold)), file = text_file)
        print("Average Payoff : " , (pay / (totalRounds - roundMinThreshold)), file = text_file)
        print("Specificty : ", (TrueNegative / (TrueNegative + FalsePositive)), file = text_file)
        print("Sensitivity : " , (TruePositive / (TruePositive + FalseNegative)), file = text_file)

        print("Final Number of Cooperators : " , (N - sum(state)), file = text_file)
        print("Number of Transitions : " , (numberOfTransitions), file = text_file)

        print("\n\n------------------------------------------------------------------------------------------------------------------------", file = text_file)


print(str(tauCenter) + " - average coop - " + str(coop / (totalRounds - roundMinThreshold)))



