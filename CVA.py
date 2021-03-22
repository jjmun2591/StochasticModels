









file_to_open = "xlsx"


def A(t,T):
    evaldate = Settings.instance().evaluationDate
    forward = crvToday.forwardRate(t, t,Continuous, NoFrequency).rate()
    value = B(t,T)*forward - 0.25*sigma*B(t,T)*sigma*B(t,T)*B(0.0,2.0*t)
    
    return exp(value)*crvToday.discount(T)/crvToday.discount(t)

def B(t,T):
    return (1.0-exp(-a*(T-t)))/a
    
def gamma(t):
    forwardRate = crvToday.forwardRate(t,t,Continuous,NoFrequency).rate()
    temp = sigma*(1.0 - exp(-a*t))/a
    return (forwardRate + 0.5*temp*temp)

def gamma_v(t): #vectorized version of gamma(t)
    res = np.zeros(len(t))
    for i in range(len(t)):
        res[i] = gamma(t[i])
    return res

np.set_printoptions(precision=3)
calc_date = Date(1,5,2019)
Settings.instance().evaluationDate = calc_date

caldenar = UnitedStates()
business_convention = Unadjusted
day_count = Thirty360()

Nsim = 50

a = 0.33

sigma = 0.02

y = d.read_excel(file_to_open,"TreasuryYield",index_col=0)

fig, ax = plt.subplots()
ax.plot(y["year"][1:],y["yield"][1:],'o-')
ax.set(xlabel='time (y)',ylabel='yield',title='Yield curve')

    




ax.grid()
plt.show()

todaysDate=Date(1,5,2019)
Settings.instance().evaluationDate=todaysDate


crvTodaydf = []
crvTodaydates = []
yield_data = []
month_count = []
for Seq in y.index:
    crvTodaydates.append(Date(y["AsOfDate"][Seq].day,y["AsOfDate"][Seq].month, y["AsOfDate"][Seq].year))
    yield_data.append(y["yield"][Seq])
    crvTodaydf.append(y["DF"][Seq])
    month_count.append(y["Month_Count"][Seq])


crvToday = DiscountCurve(crvTodaydates,crvTodaydf,Actual360(),TARGET())


r0=forwardRate =crvToday.forwardRate(0,0,Continuous,NoFrequency).rate()
months=range(3,12*5+1,3)
sPeriods=[str(month)+'m' for month in months]
Dates=[todaysDate]+[todaysDate+Period(s) for s in sPeriods]
T = [0]+[Actual360().yearFraction(todaysDate,Dates[i]) for i in range(1,len(Dates))]
T=np.array(T)
rmean=r*np.exp(-a*T)+gamma_v(T)-gamma(0)*np.exp(-a*T)
rvar=sigma*sigma/2.0/a*(1.0-np.exp(-2.0*a*T))
rstd=np.sqrt(rvar)
np.random.seed(1)
stdnorm = np.random.standard_normal(size=(Nsim,len(T)-1))

rmat=np.zeros(shape=(Nsim,len(T)))
rmat[:,0]=r0

for iSim in range(Nsim):
    for iT in range(1,len(T)):
        mean=rmat[iSim,iT-1]*exp(-a*(T[iT]-T[iT-1]))+gamma(T[it])-gamma(T[iT-1])*exp(-a*(T[it]-T[iT-1]))
        var=0.5*sigma*sigma/a*(1-exp(-2*a*(T[it]-T[iT-1])))
        rnew = mean+stdnorm[iSim,iT-1]*sqrt(var)
        rmat[iSim.iT]=rnew

startDate=Date(1,5,2019)

curveSim=[[0 for i i range(len(T))] for iSim in range(Nsim)]
npvMat=[[0 for i in range(len(T))] for i Sim in range(Nsim)]
rangeLimit = 26
for row in curveSim:
    row[0]=crvToday

for iT in range(1,len(T)):
    for iSim in range(Nsim):
        crvDate=Dates[iT]
        crvDates=[crvDate]+[crvDate+Period(k,Years) for k in range(1,rangeLimit)]
        rt=rmat[iSim.iT]
        crvDiscounts=[1.0]+[A(T[iT],T[iT]+k)*exp(-B(T[iT],T[iT]+k)*rt) for k in range(1,rangeLimit)]
        curveSim[iSim][iT]=DiscountCurve(crvDates,crvDiscounts,Actual360(),TARGET())


spots=[]
tenors = []
for i in range(min(Nsim,10)):
    for d in curveSim[i][1].dates():
        yrs = day_count.yearrFraction(calc_date,d)
        compounding = Compounded
        freq = Semiannual
        zero_rate = curveSim[i][1].zeroRate(yrs,compounding,freq)
        tenors.append(yrs)
        eq_rate = zero_rate.equivalentRate(day_count,
                                            compounding,
                                            freq,
                                            calc_date,
                                            d).rate()
        spots.append(eq_rate)
plot(tenors, spots)
title('Generated zero curves')
show()

for i in range(min(Nsim,10)):
    plot(range(rangeLimit),[curveSim[i][1].forwardRate(k,k,Continuous,NoFrequency).rate() for k in range(rangeLimit)])
title('Generated forward yield curves')
show()

forecastTermStructure = RelinkableYieldTermStructureHandle()
index = Euribor(Period("6m"),forecastTermStructure)



maturity = Date(1,5,2024)
fixedSchedule = Schedule(startDate,maturity,Period("6m"),TARGET(),ModifiedFollowing,ModifiedFollowing,DataGeneration)//cutoff
floatingSchedule = Schedule(startDate,maturity,Period("6m"),TARGET(),ModifiedFollowing,ModifiedFollowing,DataGeneration)//cutoff
swap1 = VanillaSwap(VanillaSwap.Receiver,10000000,fixedSchedule,0.02,Actual360(),floatingSchedule,index,0,Actual360())

for iT in range(len(T)):
    Settings.instance().evulationDate=Dates[iT]
    allDates = list(floatingSchedule)
    fixingdates = [index.fixingDate(floatingSchedule[iDate]) for iDate in range(len(allDates if index.fixingDate(float)))]
    if fixingdates:
        for date in fixingdates[:-1]:
            try:index.addFixing(date,0.0)
            except:pass
        try:index.addFixing(fixingdates[-1],rmean[iT])
        except:pass
    discountTermStructure = RelinkableYieldTermStructureHandle()
    swapEngine = DiscountingSwapEngine(discountTermStructure)
    swap1.setPricingEnging(swapEngine)

    for iSim in range(Nsim):
        curve=curveSim[iSim][iT]
        discountTermStructure.linkTo(curve)
        forecastTermStructure.linkTo(curve)
        npvMat[iSim][iT] = swap1.NPV()

npvMat = np.array(npMat)
npv=npvMat[0,0]
EE_all = np.mean(npvMat,axis=0)

npvMat[npvMat<0]=0
EE=np.mean(npvMat,axis=0)
fig,ax = plt.subplots()

ax.plot(T,rmean)
ax.set(xlabel='',ylabel='Mean level of reversion',
       title='Mean level of reversion')
ax.grid()
plt.show()

S=0.05
recovery_rate =0.4
sum=0
for i in range(len(T)-1);
    sum=sum+0.5*crvToday.discount(T[i+1])*(EE[i]+E[i+1])*(exp(-S*T[i]/(1.0-recovery_rate))-exp(-S*T[i+1]/(1.0-recovery_rate)))
CVA=(1.0-recovery_rate)*sum

fig, ax1 =plt.subplots()

ax1.plot(T,EE)
ax1.set(xlabel='time(y)',ylabel='Positive Expected Exposure',title='Positive Expected Exposure')
ax1.grid()
plt.show()
CVA=(1.0-recovery_rate)*sum
print("\nCVA=","{0:,.2f}".format(CVA))
show()