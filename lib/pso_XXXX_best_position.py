'''____________________________________________________________________________
#   Rev 2.0.5
#
#   XXXX Optimization with PSO  
#   James D. Kern 
#   XXXX
#   January 30, 2018
#   
#   Estimates the optimal tip XXXX location for each XXXX.
#
#   Edits:  Name\Date\Change
#           ----------------
#           James K. \ 2/8/18 \ Added loop for multiple radii
#           James K. \ 2/12/18 \ Fixed population XXXX calculation
#           James K. \ 2/16/18 \ Added "best run case" functionality
#           James K. \ 2/20/18 \ Cleaned file formatting
#
#   TODO Build softmax function for probability distribution of results
#    From Wikipedia:
#>>> import math
#>>> z = [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0]
#>>> z_exp = [math.exp(i) for i in z]
#>>> print([round(i, 2) for i in z_exp])
#[2.72, 7.39, 20.09, 54.6, 2.72, 7.39, 20.09]
#>>> sum_z_exp = sum(z_exp)
#>>> print(round(sum_z_exp, 2))
#114.98
#>>> softmax = [round(i / sum_z_exp, 3) for i in z_exp]
#>>> print(softmax)
#[0.024, 0.064, 0.175, 0.475, 0.024, 0.064, 0.175]        
#
#______________________________________________________________________________'''

'''___ IMPORT ___'''

from matplotlib import pyplot as plt
import matplotlib.animation as animation
from scipy import stats
import numpy as np
from datetime import datetime
import pylab
import random
import math
import os
import time
import sys
import logging
import warnings

warnings.filterwarnings("ignore")

'''
if os.getcwd() != 'C:\\Users\\kernj':
    os.chdir('C:\\Users\\kernj')
'''

r = np.arange(1.5,3.51,0.1)
#r = random.random(2,3)
r = list(r)
radius = []
animate = 'n'
testType = 1
inputTrials = 1
inputParticles = 1000
inputIter = 3

if animate == 'y':
    fig, ax = plt.subplots()
    plotted = False

for i in r:
    radius.append(round(i,3))

for iXXXX in [83]:
    confidence = 0
    bestRadius = 0
    bestPerSel = 0 

    for iTown in radius:
        '''___ ESTABLISH LOGGING ___'''

        #t0 = str(datetime.datetime.today().replace(microsecond=0))

        '''for char in [' ', ':']:
            if char in t0:
                t0 = t0.replace(char, '_')'''

        '''___ READ XXXX POSITION DATA ___'''

        XXXXNum = str(iXXXX)
        townDistance = iTown
        neighborRadius = 0.25

        #logging.basicConfig(format='%(message)s', runfilename='PSO Run For ' + XXXXNum + ' - Pilot' '.log', level=logging.INFO)

        #runfile = open('PSO Run For ' + XXXXNum + ' - Pilot - {}cm Radius - TESTING - FASTERTS' '.log'.format(str(townDistance)), 'w')

        #animate = input('Animation? (y/n): ')


        #runfilepath = r'C:\\Users\\kernJ' +  '\\psoXXXXs' + XXXXNum + 'full.txt'
        #runfilepath = r'Z:\\UD3\\kernj\\My Documents\\hello\\tipXXXXsXXXX82_2018-02-06_16_32_21.txt'
        fileExists = False

        while fileExists == False:
            try:
                runfilepath = 'Z:\\hello\\tipXXXXsXXXX{} - Master Set.txt'.format(XXXXNum)
                #runfilepath = 'C:\\Users\\kernj\\Documents\\Vault\\PSO\\tipXXXXsXXXX{} - Master Set.txt'.format(XXXXNum)
                #runfilepath = 'H:\\hello\\tipXXXXsXXXX{}XXXXXXXXs - Sample Data 2017.txt'.format(XXXXNum)
                runfilePos = open(runfilepath).read()
                fileExists = True
            except:
                print('Waiting for run file to be generated from tipXXXXs.py')
                time.sleep(30)
                continue

        print('On radius ' + str(townDistance) + 'cm for XXXX ' + str(XXXXNum) + '.')

        listPos = []
        listPos = runfilePos.split('\n')

        dictPos = {}
        XXXX = []
        totalXXXX = 0
        totalXXXX = 0
        keys = []
        temp = []

        for i in range(1, len(listPos) - 1):
            dictPos[i] = listPos[i]

        keys = listPos[0].split('\t')

        for i in range(1, len(dictPos) + 1):
            temp = dictPos[i].split('\t')

            dictPos[i] = {}

            for j in range(len(keys)):
                dictPos[i][keys[j]] = temp[j]

        '''___ CALCULATE TOTALS ___'''

        for i in range(1, len(dictPos) + 1):
            XXXX.append(int(dictPos[i]['XXXX']) + int(dictPos[i]['XXXX']))

        for i in range(1, len(dictPos) + 1):
            dictPos[i]['TOTALXXXX'] = XXXX[i-1]
            totalXXXX += float(dictPos[i]['XXXXXXXXXXXX'])
            totalXXXX += float(dictPos[i]['XXXXXXXXXXXX'])

        averageXXXX = totalXXXX / (len(dictPos) + 1)
        averageXXXX = totalXXXX / (len(dictPos) + 1)
        
        b1IO = []
        b1LR = []
        b2IO = []
        b2LR = []
        b3IO = []
        b3LR = []
        b4IO = []
        b4LR = []

        for i in range(1, len(dictPos) + 1):
            if dictPos[i]['TOTALXXXX'] == 0:
                b1IO.append(float(dictPos[i]['In/Out']))

                b1LR.append(float(dictPos[i]['Left/Right']))

            elif dictPos[i]['TOTALXXXX'] == 1:
                b2IO.append(float(dictPos[i]['In/Out']))

                b2LR.append(float(dictPos[i]['Left/Right']))

            elif dictPos[i]['TOTALXXXX'] == 2:
                b3IO.append(float(dictPos[i]['In/Out']))

                b3LR.append(float(dictPos[i]['Left/Right']))

            elif dictPos[i]['TOTALXXXX'] > 2:
                b4IO.append(float(dictPos[i]['In/Out']))

                b4LR.append(float(dictPos[i]['Left/Right']))

        if animate == 'y' and plotted == False:
            plt.scatter(b1LR, b1IO, s = 4, c = '#3399ff')
            plt.scatter(b2LR, b2IO, s = 4, c = '#3399ff')
            plt.scatter(b3LR, b3IO, s = 4, c = '#3399ff')
            plt.scatter(b4LR, b4IO, s = 4, c = '#3399ff')

            plt.xlabel('Left/Right')
            plt.ylabel('In/Out')

            plt.title('XXXX ' + XXXXNum + ' Best XXXX Position')
            plotted = True

        '''___ COST FUNCTION ___'''

        def XXXXS(x):
            
            score = 0
            preXXXX = 0
            postXXXX = 0
            XXXX = 0
            runScore = 0
            persel = 0
            weightPersel = 0
            weightPreXXXX = 0
            weightPostXXXX = 0
            totalScore = 1
            runXXXX = 0
            runXXXX = 0
            gamma = 0.01
            inTown = 1
            outTown = 1
            inXXXX = []
            outXXXX = []
            inXXXXTotal = 0
            inXXXXTotal = 0
            outXXXXTotal = 0
            outXXXXTotal = 0
            n1 = 0
            n2 = 0
            s1 = 0
            s2 = 0
            m1 = 0
            m2 = 0 
            t = 0
            df = 0
            p = 0
            inXCorr = []
            inYCorr = []
            town = 1
            outOfRange = 10

            for i in range(1, len(dictPos) + 1):
                dist = ((float(x[0]) - float(dictPos[i]['Left/Right']))**2 + (float(x[1]) - float(dictPos[i]['In/Out']))**2)**(0.5)

                zeroX = float(x[0])
                zeroY = float(x[1])
                quad1 = 0
                quad2 = 0
                quad3 = 0
                quad4 = 0
                #d1 = datetime.strptime(dictPos[i]['XXXXSTARTDATETIME'], "%Y-%m-%d %H:%M:%S")
                #d2 = datetime.strptime('2017-12-1', "%Y-%m-%d")
                #dayDiff = abs((d1 - d2).days)
                #relweight = 1 / (1 + dayDiff)
                relweight=1

                if  dist < float(townDistance) or dist == 0:
                    inXCorr.append(float(dictPos[i]['Left/Right']) - zeroX)
                    inYCorr.append(float(dictPos[i]['In/Out']) - zeroY)
                    inPreXXXX = int(dictPos[i]['XXXX'])
                    inPostXXXX = int(dictPos[i]['XXXX'])
                    inPersel = float(dictPos[i]['XXXX'])*relweight
                    inXXXX = float(dictPos[i]['XXXXXXXXXXXX'])*relweight
                    inXXXX = float(dictPos[i]['XXXXXXXXXXXX'])*relweight
                    XXXX = float(dictPos[i]['XXXXXXXXXXXX'])
                    inTown += 1
                    inXXXX.append(inPersel)
                    inXXXXTotal += inXXXX
                    inXXXXTotal += inXXXX

                    XXXX = float(dictPos[i]['XXXXXXXXXXXX'])

                    alpha = XXXX/averageXXXX
                    beta = XXXX/averageXXXX
                    #weightPersel = persel * weightXXXX
                    #weightPreXXXX = preXXXX * weightXXXX
                    #weightPostXXXX = postXXXX * weightXXXX

                    runScore = (1 + alpha*persel)**(1/(1 + (preXXXX + postXXXX)/alpha))
                    totalScore += runScore

                    town+=1

                else:
                    #outPreXXXX = int(dictPos[i]['XXXX'])
                    #outPostXXXX = int(dictPos[i]['XXXX'])
                    outPersel = float(dictPos[i]['XXXX'])*relweight
                    outXXXX = float(dictPos[i]['XXXXXXXXXXXX'])*relweight
                    outXXXX = float(dictPos[i]['XXXXXXXXXXXX'])*relweight
                    outTown += 1
                    outXXXX.append(outPersel)
                    outXXXXTotal += outXXXX
                    outXXXXTotal += outXXXX

            runScore = (1/(totalScore/town))/(town**gamma)

            n1 = inTown
            n2 = outTown

            s1 = np.std(inXXXX)
            s2 = np.std(outXXXX)

            try:
                inXXXX = inXXXXTotal / inXXXXTotal
                outXXXX = outXXXXTotal / outXXXXTotal
            except:
                inXXXX = 0
                outXXXX = 0

            m1 = inXXXX
            m2 = outXXXX

            t = abs(m1-m2)/(np.sqrt(s1**2/n1+s2**2/n2))

            df = n1+n2-2
            p = 1 - stats.t.cdf(t,df=df)

            tempConfidence = 100 - 2*p*100
            nConfidence = (1/(1 + tempConfidence))

            '''if m1 < m2:
                nConfidence = outOfRange'''

            for i in range(len(inXCorr)):
                if inXCorr[i] > 0 and inYCorr[i] > 0:
                    quad1+=1
                elif inXCorr[i] > 0 and inYCorr[i] < 0:
                    quad2+=1
                elif inXCorr[i] < 0 and inYCorr[i] < 0:
                    quad3+=1
                elif inXCorr[i] < 0 and inYCorr[i] > 0:
                    quad4+=1
            
            if quad1 < .15*inTown:
                nConfidence = outOfRange

            if quad2 < .15*inTown:
                nConfidence = outOfRange

            if quad3 < .15*inTown:
                nConfidence = outOfRange

            if quad4 < .15*inTown:
                nConfidence = outOfRange

            if inTown < .05*outTown:
                nConfidence = outOfRange

            if testType == 1:
                return nConfidence
            elif testType == 2:
                return runScore

        '''___ MAIN PSO ___'''

        class Particle:
            def __init__(self, X, numDim):
                self.dim = numDim
                self.errIndBest = -999
                self.errIndividual = -999
                self.positionIndividual = []
                self.velocityIndividual = []
                self.posIndBest = []

                for i in range(0, numDim): # Intitialize random velocities and postions for each particle
                    self.velocityIndividual.append(random.uniform(-1, 1))
                    self.positionIndividual.append(X[i])

            def update_velocity_position(self, posGroupBest, bounds):
                w, c1, c2 = .8, 0.1, 0.1     # weights
                for i in range(0, self.dim):
                    # Update velocity
                    r = np.random.uniform(0, 1, 1)[0]
                    # V term is a combination of neighborhood and individual particle effects
                    V = c1 * r * (self.posIndBest[i] - self.positionIndividual[i]) + c2 * r * (posGroupBest[i] - self.positionIndividual[i])
                    self.velocityIndividual[i] = w * self.velocityIndividual[i] + V

                    # Update position
                    self.positionIndividual[i] = self.positionIndividual[i] + self.velocityIndividual[i]
                    if self.positionIndividual[i] > bounds[i][1]:
                        self.positionIndividual[i] = bounds[i][1]
                    elif self.positionIndividual[i] < bounds[i][0]:
                        self.positionIndividual[i] = bounds[i][0]

            def evaluate_cost(self,costFunc):
                self.errIndividual = costFunc(self.positionIndividual)
                if self.errIndividual < self.errIndBest or self.errIndBest == -1:
                    self.errIndBest = self.errIndividual # Assign new best error
                    self.posIndBest = self.positionIndividual # Assign new best position

        def PSO_Animation(costFunc, X, bounds, numParticles, maxIter, b):
            numDim = len(X)
            bestGroupError = -999
            posGroupBest = [0,0]                   
            bestX, bestY = posGroupBest[0], posGroupBest[1]

            # Initializing the list of particles (see initialized attributes above)
            swarm = []
            for i in range(0, numParticles):
                X = [random.uniform(-5, 5), random.uniform(-5, 5)]
                swarm.append(Particle(X, numDim))

            if animate == 'y':
                new_x = [i.positionIndividual[0] for i in swarm]
                new_y = [i.positionIndividual[1] for i in swarm]
                fig = plt.style.use('dark_background')
                points, = ax.plot(new_x, new_y, marker='8', linestyle='None', markersize=1, c = '#ff66ff')
                best, = ax.plot(bestX, bestY, marker='x', c='#99ff33', markersize=9)
                ax.set_facecolor('#666666')
                ax.set_xlim(-b, b)
                ax.set_ylim(-b, b)

            # Particle swarm optimization iteration
            i = 0
            while i < maxIter:
                new_x = [i.positionIndividual[0] for i in swarm]
                new_y = [i.positionIndividual[1] for i in swarm]

                if animate == 'y':
                    points.set_data(new_x, new_y) # Points for animation
                    if i > 1:
                        best.set_data(bestX, bestY)

                # Determine global minimum
                for j in range(0, numParticles):
                    swarm[j].evaluate_cost(costFunc)
                    if swarm[j].errIndividual < bestGroupError or bestGroupError == -1:
                        posGroupBest = list(swarm[j].positionIndividual)
                        bestX = posGroupBest[0]
                        bestY = posGroupBest[1] 
                        bestGroupError = float(swarm[j].errIndividual)
                        index_best = j

                # Update velocities and positions before next iteration
                for j in range(0, numParticles):
                    swarm[j].update_velocity_position(posGroupBest, bounds)

                i += 1
                plt.pause(0.0001)

            if animate == 'y':
                points.remove()
                best.remove()
                plt.scatter(posGroupBest[0], posGroupBest[1], s = 100, c = 'r', marker = "x")

            return (posGroupBest[0], posGroupBest[1], index_best, bestGroupError)
        

        def doPSO():
            global numTrials, numParticles, maxIter
            numTrials = inputTrials
            numParticles = inputParticles
            maxIter = inputIter

            bestPos = []
            bestIndex = []
            b = 7

            for i in range(numTrials):
                #print('On simulation ' + str(i+1) + ' out of ' + str(numTrials) + '.')
    
                init_pos = (random.uniform(-b, b), random.uniform(-b, b))
                initial = [*init_pos]                                       # Initial starting location [(In/Out),(Left/Right)]

                #print(f'Initial particle position: <{init_pos[0]},{init_pos[1]}>')
                #logging.info(f'Starting simluation #{i+1}')

                bounds = [(-b,b),(-b,b)]                                    # Input bounds [(x1min, x1max), (x2min, x2max)]

                #PSO(XXXXS, initial, bounds, numParticles = 15, maxIter = 40)

                xy = PSO_Animation(XXXXS, initial, bounds, numParticles, maxIter, b)

                bestPos.append(xy[:2])
                bestIndex.append(xy[2])

            return bestPos

        '''____ DETERMINE BEST POSITION ____'''
        
        bestPos = doPSO()

        finalPos = {}

        for i in range(len(bestPos)):
            tempPos = bestPos[i]
            finalPos[i] = {}
            finalPos[i]['xy'] = tempPos
            count = 0
            neighbor = []
            for j in range(len(bestPos)):
                if ((tempPos[0] - bestPos[j][0])**2 + (tempPos[1] - bestPos[j][1])**2)**(0.5) < neighborRadius:
                    count += 1
                    neighbor.append(j)
            finalPos[i]['finalCount'] = count
            finalPos[i]['neighbors'] = neighbor

        maxFinalCount = 0

        for i in range(len(finalPos)):
            if finalPos[i]['finalCount'] > maxFinalCount:
                maxFinalCount = finalPos[i]['finalCount']
                indexMax = i



        '''____ FIND XXXX IN REGION ____'''

        runBestTown = 0
        bestXXXX = 0
        bestPreBreak = 0
        bestPostBreak = 0
        bestXXXX = 0
        bestXXXX = 0
        runRemainTown = 0
        remainXXXX = 0
        remainPreBreak = 0
        remainPostBreak = 0
        remainXXXX = 0
        remainXXXX = 0
        testDist = 0
        stdRunBestXXXX = []
        stdRunRemainXXXX = []
        tTestBestXXXX = []
        tTestRemainXXXX = []
        testingIOIndex = []
        testingLRIndex = []
        listFinalX = []
        listFinalY = []
        listConfidence = []
        listPerSel = []


        for i in range(1, len(dictPos) + 1): 
            testDist = ((float(dictPos[i]['Left/Right']) - float(finalPos[indexMax]['xy'][0]))**2 + \
                        (float(dictPos[i]['In/Out']) - float(finalPos[indexMax]['xy'][1]))**2)**(0.5)

            if testDist < float(townDistance):
                bestXXXX += float(dictPos[i]['XXXX'])
                bestPreBreak += float(dictPos[i]['XXXX'])
                bestPostBreak += float(dictPos[i]['XXXX'])
                stdRunBestXXXX.append(float(dictPos[i]['XXXX']))
                tTestBestXXXX.append(float(dictPos[i]['XXXX']))
                bestXXXX += float(dictPos[i]['XXXXXXXXXXXX'])
                bestXXXX += float(dictPos[i]['XXXXXXXXXXXX'])
                runBestTown += 1
                testingIOIndex.append(float(dictPos[i]['In/Out']))
                testingLRIndex.append(float(dictPos[i]['Left/Right']))
                dictPos.pop(i, None)                                    # Remove from dictionary to prepare for calculating mean XXXX of remainder population

        for i in range(1, len(dictPos) + runBestTown + 1):
            try:
                #if (float(dictPos[i]['In/Out'])**2 + float(dictPos[i]['Left/Right'])**2)**(0.5) < 7:
                    remainXXXX += float(dictPos[i]['XXXX'])
                    remainPreBreak += float(dictPos[i]['XXXX'])
                    remainPostBreak += float(dictPos[i]['XXXX'])
                    stdRunRemainXXXX.append(float(dictPos[i]['XXXX']))
                    tTestRemainXXXX.append(float(dictPos[i]['XXXX']))
                    remainXXXX += float(dictPos[i]['XXXXXXXXXXXX'])
                    remainXXXX += float(dictPos[i]['XXXXXXXXXXXX'])
                    runRemainTown += 1
            except:
                continue
        try:
            runImprovedXXXX = (bestXXXX/bestXXXX - ((remainXXXX + bestXXXX)/(remainXXXX + bestXXXX)))*100
            runPerSel = (bestXXXX/bestXXXX)*100
            runRemainPerSel = (remainXXXX/remainXXXX)*100
            runPopPerSel = ((remainXXXX + bestXXXX)/(remainXXXX + bestXXXX))*100
        except:
            runImprovedXXXX = 0

        '''____ T-TEST ____'''

        n1 = len(tTestBestXXXX)
        n2 = len(tTestRemainXXXX)

        s1 = np.std(tTestBestXXXX)
        s2 = np.std(tTestRemainXXXX)

        m1 = runPerSel
        m2 = runRemainPerSel

        t = abs(m1-m2)/(np.sqrt(s1**2/n1+s2**2/n2))

        df = n1+n2-2
        p = 1 - stats.t.cdf(t,df=df)

        runConfidence = 100 - 2*p*100

        '''____ RUN RESULTS ____'''

        '''
        runfile.write('Best position for ' + XXXXNum + ' is within ' + str(townDistance) + 'cm of (' + str('{:.3f}'.format(finalPos[indexMax]['xy'][0])) + ', ' +\
        str('{:.3f}'.format(finalPos[indexMax]['xy'][1])) + ').')

        runfile.write('\n\nYou can be {}% confident that this region will provide {}% improved XXXX.\n'.format(str('{:.3f}'.format(runConfidence)), str('{:.3f}'.format(runImprovedXXXX))))

        runfile.write('\nNumber of simulations: ' + str(numTrials))
        runfile.write('\nNumber of particles:   ' + str(numParticles))
        runfile.write('\nNumber of iterations:  ' + str(maxIter) + '\n\n')

        runfile.write(str(float('{:.3f}'.format(runPopPerSel))) + f'% population XXXX.\n\n')

        #logging.info('Cost function: \n' + scoreString)

        runfile.write(str(float('{:.3f}'.format(runPerSel))) + f'% best XXXX.\n')
        runfile.write(str(runBestTown) + ' number of XXXX.\n')
        runfile.write('{:.3f}'.format(np.std(stdRunBestXXXX)) + ' S.D. best XXXX.\n\n')

        runfile.write(str(float('{:.3f}'.format(runRemainPerSel))) + f'% remain XXXX.\n')
        runfile.write(str(runRemainTown) + ' number of XXXX.\n')
        runfile.write('{:.3f}'.format(np.std(stdRunRemainXXXX)) + ' S.D. remain XXXX.\n\n')

        runfile.write('Your t-statistic: ' + str('{:.4f}'.format(t)))
        runfile.write('\nP-value:          ' + str('{:.4f}'.format(2*p)))

        runfile.close()

        #plt.scatter(finalPos[indexMax]['xy'][0], finalPos[indexMax]['xy'][1], s = 150, c = '#ffcc00', marker = "x")

        #plt.show()'''

        #if runConfidence > confidence and runPerSel > bestPerSel:
            #if townDistance > bestRadius:

        
        if runPerSel > bestPerSel and runConfidence > 95:
            confidence = runConfidence
            bestRadius = townDistance
            bestPerSel = runPerSel
            remainPerSel = runRemainPerSel
            stdBestXXXX = stdRunBestXXXX
            stdRemainXXXX = stdRunRemainXXXX
            improvedXXXX = runImprovedXXXX
            bestPopPerSel = runPopPerSel
            bestTrials = numTrials
            bestParticles = numParticles
            bestIter = maxIter
            bestIndex = indexMax
            bestTown = runBestTown
            remainTown = runRemainTown
            finalIOIndex = testingIOIndex
            finalLRIndex = testingLRIndex
            finalPositionLR = str('{:.3f}'.format(finalPos[bestIndex]['xy'][0]))
            finalPositionIO = str('{:.3f}'.format(finalPos[bestIndex]['xy'][1]))
            listFinalX.append(float(finalPositionLR))
            listFinalY.append(float(finalPositionIO))
            listConfidence.append(runConfidence)
            listPerSel.append(runPerSel)

    '''____ BEST RUN RESULTS & LOGFILE ____'''

    # TODO: How to print to console and logrunfile same time ???
    
    file = open('Best Confidence Run For ' + XXXXNum + ' - Sample Data 2017 - XXXX End Code 699 {}cm Radius' '.log'.format(str(bestRadius)), 'w')
    file2 = open('Best Confidence Run For ' + XXXXNum + ' - Sample Data 2017 - XXXX End Code 699 {}cm Radius - XXXXs' '.log'.format(str(bestRadius)), 'w')

    for i in range(len(finalIOIndex)):
        file2.write(str(finalIOIndex[i]) + '\t' + str(finalLRIndex[i]) + '\n')

    print('\n' + str(float('{:.3f}'.format(bestPerSel))) + f'% best XXXX.')
    print(str(bestTown) + ' number of XXXX.')
    print('{:.3f}'.format(np.std(stdBestXXXX)) + ' S.D. best XXXX.\n')

    print(str(float('{:.3f}'.format(remainPerSel))) + f'% remain XXXX.')
    print(str(remainTown) + ' number of XXXX.')
    print('{:.3f}'.format(np.std(stdRemainXXXX)) + ' S.D. remain XXXX.\n')

    print(str(float('{:.3f}'.format(bestPopPerSel))) + f'% population XXXX.\n')

    print('Your t-statistic: ' + str('{:.4f}'.format(t)))
    print('P-value:          ' + str('{:.4f}'.format(2*p)))

    print('\nBest position for ' + XXXXNum + ' is within ' + str(bestRadius) + 'cm of ' + finalPositionLR + ',' + finalPositionIO + '.')

    print('\nYou can be {}% confident that this region will provide {}% improved XXXX.\n'.format(str('{:.3f}'.format(confidence)), str('{:.3f}'.format(improvedXXXX))))

    file.write('Best position for ' + XXXXNum + ' is within ' + str(bestRadius) + 'cm of ' + finalPositionLR + ',' + finalPositionIO + '.')

    file.write('\n\nYou can be {}% confident that this region will provide {}% improved XXXX.\n'.format(str('{:.3f}'.format(confidence)), str('{:.3f}'.format(improvedXXXX))))

    file.write('\nNumber of simulations: ' + str(bestTrials))
    file.write('\nNumber of particles:   ' + str(bestParticles))
    file.write('\nNumber of iterations:  ' + str(bestIter) + '\n\n')

    file.write(str(float('{:.3f}'.format(bestPopPerSel))) + f'% population XXXX.\n\n')

    file.write(str(float('{:.3f}'.format(bestPerSel))) + f'% best XXXX.\n')
    file.write(str(bestTown) + ' number of XXXX.\n')
    file.write('{:.3f}'.format(np.std(stdBestXXXX)) + ' S.D. best XXXX.\n\n')

    file.write(str(float('{:.3f}'.format(remainPerSel))) + f'% remain XXXX.\n')
    file.write(str(remainTown) + ' number of XXXX.\n')
    file.write('{:.3f}'.format(np.std(stdRemainXXXX)) + ' S.D. remain XXXX.\n\n')

    file.write('Your t-statistic: ' + str('{:.4f}'.format(t)))
    file.write('\nP-value:          ' + str('{:.4f}'.format(2*p)))

    file.close()

    #plt.scatter(finalPos[indexMax]['xy'][0], finalPos[indexMax]['xy'][1], s = 150, c = '#ffcc00', marker = "x")

    #plt.show()

    '''____ END ____'''

