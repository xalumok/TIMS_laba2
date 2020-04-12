import numpy as np
from tkinter import *
import matplotlib.pyplot as plt
import random
import math
from collections import Counter
import plotly.graph_objects as go
import tkinter.font as font

def getCntNew(a):
    elems = np.unique(a)
    res = []
    for i in range(len(elems)):
        res.append([elems[i], a.count(elems[i])])
    return res


def getCnt(a):
    elems = np.unique(a)
    cnts = [a.count(elem) for elem in elems]
    return [elems, cnts]


def getIntervalCnt(a):
    r = 1
    for i in range(10000):
        if 2 ** i < len(a) <= 2 ** (i + 1):
            r = i
            break
    intervalCnt = r
    mn = min(a)
    mx = max(a)
    intervalLen = (mx - mn) / intervalCnt
    intervals = []
    cnts = []
    for i in range(intervalCnt):
        l = mn + i * intervalLen
        r = mn + (i + 1) * intervalLen
        l = round(l, 2)
        r = round(r, 2)
        curInterval = [l, r]
        if i == 0:
            cnt = sum(1 for k in a if l <= k <= r)
        else:
            cnt = sum(1 for k in a if l < k <= r)
        intervals.append(curInterval)
        cnts.append(cnt)
    return [[str(elem) for elem in intervals], cnts]


def getIntervalCntNew(a):
    r = 1
    for i in range(10000):
        if 2 ** i < len(a) <= 2 ** (i + 1):
            r = i
            break
    intervalCnt = r
    mn = min(a)
    mx = max(a)
    intervalLen = (mx - mn) / intervalCnt
    res = []
    for i in range(intervalCnt):
        l = mn + i * intervalLen
        r = mn + (i + 1) * intervalLen
        l = round(l, 2)
        r = round(r, 2)
        curInterval = [l, r]
        if i == 0:
            cnt = sum(1 for k in a if l <= k <= r)
        else:
            cnt = sum(1 for k in a if l < k <= r)
        res.append([curInterval, round(mn + (i+0.5) * intervalLen,2), cnt])

    return res


def show_bar(inputList):
    inp = getCnt(inputList)
    plt.figure()
    plt.ylabel('count')
    plt.xlabel('x')
    plt.bar(inp[0], inp[1], width=0.2)
    plt.show()


def show_plot(inputList):
    inp = getCnt(inputList)
    plt.figure()
    plt.ylabel('count')
    plt.xlabel('x')
    plt.plot(inp[0], inp[1])
    plt.show()


def show_hist(inputList):
    inp = getIntervalCnt(inputList)
    plt.figure()
    plt.ylabel('count')
    plt.xlabel('x')
    plt.bar(inp[0], inp[1], width=0.99)
    plt.show()


def show_emp(inputList):
    t = getCnt(inputList)
    elems = [t[0][0] - 5]
    elems.extend(t[0])
    elems.append(elems[-1] + 5)
    cnts = [0]
    cnts.extend(t[1])
    cnts.append(cnts[-1])
    g = []
    cntSum = 0
    for i in range(len(elems) - 1):
        cntSum += cnts[i]
        g.append([[elems[i], cntSum], [elems[i + 1], cntSum]])
        plt.plot((elems[i], elems[i+1]), ((cntSum-cnts[i]) / len(inputList), (cntSum) / len(inputList)), color='green')
    plt.show()

def show_emp2(inputList):
    t = getCnt(inputList)
    elems = [t[0][0] - 5]
    elems.extend(t[0])
    elems.append(elems[-1] + 5)
    cnts = [0]
    cnts.extend(t[1])
    cnts.append(cnts[-1])
    g = []
    cntSum = 0
    for i in range(len(elems) - 1):
        cntSum += cnts[i]
        g.append([[elems[i], cntSum], [elems[i + 1], cntSum]])
        plt.plot((elems[i], elems[i + 1]), (cntSum / len(inputList), cntSum / len(inputList)), color='green')
    plt.show()


def getQuant(a):
    n = len(a)
    res = []
    if n % 4 == 0:
        for i in range(1, 4):
            res.append(["Q" + str(i), str(a[(n // 4) * i - 1])])
        res.append(["Q3 - Q1", str(a[(3 * n) // 4 - 1] - a[n // 4 - 1])])
    if n % 10 == 0:
        for i in range(1, 10):
            res.append(["D" + str(i), str(a[(n // 10) * i - 1])])
        res.append(["D9 - D1", str(a[(9 * n) // 10 - 1] - a[n // 10 - 1])])
    if n % 100 == 0:
        for i in range(1, 100):
            res.append(["C" + str(i), str(a[(n // 100) * i - 1])])
        res.append(["C99 - C1", str(a[(99 * n) // 100 - 1] - a[n // 100 - 1])])
    if n % 1000 == 0:
        for i in range(1, 1000):
            res.append(["M" + str(i / 10), str(a[(n // 1000) * i - 1])])
        res.append(["M99,9 - M0,1", str(a[(999 * n) // 1000 - 1] - a[n // 1000 - 1])])
    return res


def getMomentums(a):
    n = len(a)
    res = []
    for i in range(0, 5):
        res.append(["Momentum(" + str(i) + ")", sum([elem ** i for elem in a]) / n])
    return res


def getCentralMomentums(a, mean):
    n = len(a)
    res = []
    for i in range(1, 5):
        res.append(["Central Momentum(" + str(i) + ")", sum([(elem - mean) ** i for elem in a]) / n])
    res.append(["Asymetry", res[2][1] / (res[1][1] ** 1.5)])
    res.append(["Excess", res[3][1] / (res[1][1] ** 2) - 3])
    return res


def getInfo(inputList):
    inputList.sort()
    n = len(inputList)
    lst = np.array(inputList)
    res = []
    res.append(["Median", np.median(lst)])
    res.append(["Mode", Counter(lst).most_common(1)[0][0]])
    lstMean = np.mean(lst)
    res.append(["Mean", lstMean])
    res.append(["Scope", lst.max() - lst.min()])
    lstDev = sum([(x - lstMean) ** 2 for x in lst])
    res.append(["Deviation", lstDev])
    lstVariance = lstDev / (n - 1)
    res.append(["Variance", lstVariance])
    lstStd = math.sqrt(lstVariance)
    res.append(["Standart", lstStd])
    res.append(["Dispersion", lstDev / n])
    res.append(["Variation", lstStd / lstMean])
    res.extend(getQuant(inputList))
    res.extend(getMomentums(inputList))
    res.extend(getCentralMomentums(inputList, lstMean))
    return res


def getIntervalMode(a):
    ind = 0
    mx = 0
    for i in range(len(a)):
        if a[i][2]>mx:
            mx = a[i][2]
            ind = i
    prevCnt = 0
    if ind != 0:
        prevCnt = a[ind-1][2]
    nextCnt = 0
    if ind != len(a)-1:
        nextCnt = a[ind+1][2]
    res = a[ind][0][0] + (a[ind][0][1]-a[ind][0][0]) * (a[ind][2] - prevCnt) / (a[ind][2] - prevCnt + a[ind][2]+nextCnt)
    return res

def getIntervalMedian(a):
    ind = 0
    mx = 0
    sumCnt = 0
    sumToMax = 0
    print(a)
    for i in range(len(a)):
        if a[i][2]>mx:
            mx = a[i][2]
            ind = i
        sumCnt += a[i][2]
    for i in range(ind):
        sumToMax += a[i][2]
    res = a[ind][0][0] + (a[ind][0][1]-a[ind][0][0]) * (sumCnt/2 - sumToMax) / a[ind][2]
    return res

def getIntervalMomentums(a,allCnt):
    n = len(a)
    res = []
    for i in range(0, 5):
        res.append(["Momentum(" + str(i) + ")", sum([a[j][2] * (a[j][1] ** i) for j in range(n)]) / allCnt])
    return res


def getIntervalCentralMomentums(a, mean, allCnt):
    n = len(a)
    res = []
    for i in range(1, 5):
        res.append(["Central Momentum(" + str(i) + ")", sum([a[j][2] * ((a[j][1] - mean) ** i) for j in range(n)]) / allCnt])
    res.append(["Asymetry", res[2][1] / (res[1][1] ** 1.5)])
    res.append(["Excess", res[3][1] / (res[1][1] ** 2) - 3])
    return res


def getIntervalInfo(inputList):
    n = len(inputList)
    Zi = []
    allCnt = 0
    Ni = []
    for i in range(n):
        Zi.append(inputList[i][1])
        Ni.append(inputList[i][2])
        allCnt += Ni[i]
    res = []
    res.append(["Median", getIntervalMedian(inputList)])
    res.append(["Mode", getIntervalMode(inputList)])
    lstMean = sum([Zi[u]*Ni[u] for u in range(n)])/allCnt
    res.append(["Mean", lstMean])
    lstDev = sum([Ni[j] *((Zi[j] - lstMean)** 2)  for j in range(n)])
    res.append(["Deviation", lstDev])
    lstVariance = lstDev / (allCnt - 1)
    res.append(["Variance", lstVariance])
    lstStd = math.sqrt(lstVariance)
    res.append(["Standart", lstStd])
    res.append(["Dispersion", lstDev / allCnt])
    res.append(["Variation", lstStd / lstMean])
    res.extend(getIntervalMomentums(inputList, allCnt))
    res.extend(getIntervalCentralMomentums(inputList, lstMean, allCnt))
    return res

def generate(mn, mx, cnt):
    return [math.floor(random.uniform(mn, mx)) for i in range(cnt)]


def readFromFile(fileName):
    try:
        result = []
        file = open(fileName)
        inp = file.readlines()
        for line in inp:
            x, n = map(int, line.split())
            result.extend([x for i in range(n)])
        return result
    except:
        print('ERROR')
        messagebox.showerror("ERROR", "Invalid file")


###########################
class Data:
    def init(self):
        self.lst = []

    def getter(self):
        return self.lst


data = Data()
window = Tk()
window.title('TIMS_1')
window.geometry("610x180")
myFont = font.Font(family='Courier', size=10, weight='bold')

lb = Label(window, text='Input TextFile Path: ')
lb.grid(row=0, column=0)

fileName_inp = Entry(window)
fileName_inp.grid(row=0, column=1)


def fileInput():
    fileName = fileName_inp.get()
    data.lst = readFromFile(fileName)


textfile_bt = Button(window, text='Input from file', command=fileInput)
textfile_bt.grid(row=0, column=2)

def generateList():
    mn = float(border_left_entry.get())
    mx = float(border_right_entry.get())
    cnt = int(count_entry.get())
    data.lst = generate(mn, mx, cnt)

button = Button(window, text='Generate', command=generateList)
button.grid(row=5, column=1)

lb_border_left = Label(window, text='Min: ')
lb_border_left.grid(row=2, column=0)

border_left_entry = Entry(window)
border_left_entry.grid(row=2, column=1)

lb_border_right = Label(window, text='Max: ')
lb_border_right.grid(row=3, column=0)

border_right_entry = Entry(window)
border_right_entry.grid(row=3, column=1)

lb_count = Label(window, text='N: ')
lb_count.grid(row=4, column=0)

count_entry = Entry(window)
count_entry.grid(row=4, column=1)


def Dia_chastot():
    show_bar(data.lst)


dia_chastot_but = Button(window, text='Frequency Diagram', command=Dia_chastot)
dia_chastot_but.grid(row=3, column=5)


def Pol_chastot():
    show_plot(data.lst)


pol_chastot_but = Button(window, text='Frequency Ground', command=Pol_chastot)
pol_chastot_but.grid(row=4, column=5)


def His_chastot():
    show_hist(data.lst)


his_chastot_but = Button(window, text='Frequency Histogram', command=His_chastot)
his_chastot_but.grid(row=5, column=5)


def Tables():

    arr = getCntNew(data.lst)
    headers = [["Xi", "Ni"]]
    headers.extend(arr)
    generateWindowWithInfo(headers)


table_but = Button(window, text='Tables', command=Tables)
table_but.grid(row=0, column=5)


def Interval_Tables():
    arr = getIntervalCntNew(data.lst)
    headers = [["[Zi, Zi+1]", "Z", "Ni"]]
    headers.extend(arr)
    generateWindowWithInfo(headers)


table_but = Button(window, text='Interval Tables', command=Interval_Tables)
table_but.grid(row=1, column=5)


def generateWindowWithInfo(arr):
    output_window = Tk()
    output_window.title('Info')
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            Label(output_window, text=str(arr[i][j])).grid(row=i, column=j)


def info():
    inf = getInfo(data.lst)
    for i in range(len(inf)):
        inf[i][1] = round(float(inf[i][1]), 2)
    generateWindowWithInfo(inf)


discrete_info_but = Button(window, text='Charasterictics', command=info)
discrete_info_but.grid(row = 3, column = 4)

def interval_info():

    inf = getIntervalInfo(getIntervalCntNew(data.lst))
    for i in range(len(inf)):
        inf[i][1] = round(float(inf[i][1]), 2)
    generateWindowWithInfo(inf)

interval_info_but = Button(window, text = 'Interval Charasterictics', command = interval_info)
interval_info_but.grid(row = 4, column = 4)

def emp():
    show_emp(data.lst)


emp_but = Button(window, text='Empirical Distribution Function Interval', command=emp)
emp_but.grid(row=6, column=5)

def emp2():
    show_emp2(data.lst)
emp_but2 = Button(window, text='Empirical Distribution Function Discrete', command=emp2)
emp_but2.grid(row=6, column=4)


################
def generateUniformDistr(elemCnt, n):
	lst = []
	for i in range(n):
		lst.append([str(round(n/elemCnt,2)), str(round(elemCnt / n,2))])
	return lst

def discr_uniform():
	arr = getCntNew(data.lst)
	ind = 0
	r = len(arr)-1
	for elem in generateUniformDistr(len(data.lst), len(arr)):
		arr[ind].append(elem[0])
		arr[ind].append(elem[1])
		ind += 1
	headers = [["x_i", "m_i", "p_i", "n*p_i"]]
	headers.extend(arr)
	x2_emp = 0
	for i in range(1, len(arr)):
		x2_emp += ((float(arr[i][1])-float(arr[i][3]))**2) / float(arr[i][3])

	headers.append(["X2_emp","","",str(round(x2_emp, 2))])
	alpha = list(open("table.txt").readlines())
	x2_crit = float(alpha[r].replace(",", "."))
	headers.append(["X2_crit","","",str(alpha[r])])
	headers.append(["Result:","","", ("Hipothesis is right" if x2_emp <= x2_crit else "Hipothesis is wrong")])
	generateWindowWithInfo(headers)
###############

def generateUniformDistrInterval(elemCnt, n):
	lst = []
	for i in range(n):
		lst.append([str(round(1/n,2)), str(round(elemCnt / n,2))])
	return lst

def interval_uniform():
	arr = getIntervalCntNew(data.lst)
	for elem in arr:
		elem[0] = str(elem[0])
		elem.pop(1)
	ind = 0
	r = len(arr)-1
	for elem in generateUniformDistrInterval(len(data.lst), len(arr)):
		arr[ind].append(elem[0])
		arr[ind].append(elem[1])
		ind += 1
	headers = [["[l, r]", "m_i", "p_i", "n*p_i"]]
	headers.extend(arr)
	x2_emp = 0
	for i in range(1, len(arr)):
		x2_emp += ((float(arr[i][1])-float(arr[i][3]))**2) / float(arr[i][3])

	headers.append(["X2_emp","","",str(round(x2_emp, 2))])
	alpha = list(open("table.txt").readlines())
	x2_crit = float(alpha[r].replace(",", "."))
	headers.append(["X2_crit","","",str(alpha[r])])
	headers.append(["Result:","","", ("Hipothesis is right" if x2_emp <= x2_crit else "Hipothesis is wrong")])
	generateWindowWithInfo(headers)

discr_uniform_but = Button(window, text='Discrete Uniform Distribution', command=discr_uniform)
discr_uniform_but.grid(row=8, column=4)

interv_uniform_but = Button(window, text='Interval Uniform Distribution', command=interval_uniform)
interv_uniform_but.grid(row=9, column=4)

###################

################

def C(n, k):
	return (math.factorial(n) / (math.factorial(k)*math.factorial(n-k)))

def getBin(n, k, p):
	return (p**k)*((1-p)**(n-k))*C(n, k)

def generateBinDistr(elemCnt, n):
	lst = []
	for i in range(n):
		lst.append([str(round(getBin(n, i, 0.5),2)), str(round(elemCnt * getBin(n, i, 0.5),2))])
	return lst

def discr_bin():
	arr = getCntNew(data.lst)
	ind = 0
	r = len(arr)-1
	for elem in generateBinDistr(len(data.lst), len(arr)):
		arr[ind].append(elem[0])
		arr[ind].append(elem[1])
		ind += 1
	headers = [["x_i", "m_i", "p_i", "n*p_i"]]
	headers.extend(arr)
	x2_emp = 0
	for i in range(1, len(arr)):
		x2_emp += ((float(arr[i][1])-float(arr[i][3]))**2) / float(arr[i][3])

	headers.append(["X2_emp","","",str(round(x2_emp, 2))])
	alpha = list(open("table.txt").readlines())
	x2_crit = float(alpha[r].replace(",", "."))
	headers.append(["X2_crit","","",str(alpha[r])])
	headers.append(["Result:","","", ("Hipothesis is right" if x2_emp <= x2_crit else "Hipothesis is wrong")])
	generateWindowWithInfo(headers)
###############

def generateBinInterval(elemCnt, n):
	lst = []
	for i in range(n):
		lst.append([str(round(getBin(n, i, 0.5),2)), str(round(elemCnt * getBin(n, i, 0.5),2))])
	return lst

def interval_bin():
	arr = getIntervalCntNew(data.lst)
	for elem in arr:
		elem[0] = str(elem[0])
		elem.pop(1)
	ind = 0
	r = len(arr)-1
	for elem in generateBinInterval(len(data.lst), len(arr)):
		arr[ind].append(elem[0])
		arr[ind].append(elem[1])
		ind += 1
	headers = [["[l, r]", "m_i", "p_i", "n*p_i"]]
	headers.extend(arr)
	x2_emp = 0
	for i in range(1, len(arr)):
		x2_emp += ((float(arr[i][1])-float(arr[i][3]))**2) / float(arr[i][3])

	headers.append(["X2_emp","","",str(round(x2_emp, 2))])
	alpha = list(open("table.txt").readlines())
	x2_crit = float(alpha[r].replace(",", "."))
	headers.append(["X2_crit","","",str(alpha[r])])
	headers.append(["Result:","","", ("Hipothesis is right" if x2_emp <= x2_crit else "Hipothesis is wrong")])
	generateWindowWithInfo(headers)

discr_bin_but = Button(window, text='Discrete Binomial Distribution', command=discr_bin)
discr_bin_but.grid(row=10, column=4)

interv_bin_but = Button(window, text='Interval Binomial Distribution', command=interval_bin)
interv_bin_but.grid(row=11, column=4)
window.geometry("800x300")
window.mainloop()