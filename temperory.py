print(c_neww['MEAN'].head())
print(c_neww.info())
print('...')
print(neww.info())
print(neww)
print('...')
sim_2 = neww.iloc[:, 1:2];
sim_3 = neww.iloc[:, 2:3];
sim_4 = neww.iloc[:, 3:4];
sim_5 = neww.iloc[:, 4:5]
sim_6 = neww.iloc[:, 5:6];
sim_7 = neww.iloc[:, 6:7];
sim_8 = neww.iloc[:, 7:8];
sim_9 = neww.iloc[:, 8:9]
ax[0][0].plot(neww['Actual Nestle Close Price'], color='b')
ax[0][0].plot(c_neww['MEAN'], color='r')
ax[0][1].plot(neww['Actual Nestle Close Price'], color='b')
ax[0][1].plot(sim_2, color='r')
ax[0][2].plot(neww['Actual Nestle Close Price'], color='b')
ax[0][2].plot(sim_3, color='r')
ax[1][0].plot(neww['Actual Nestle Close Price'], color='b')
ax[1][0].plot(sim_4, color='r')
ax[1][1].plot(neww['Actual Nestle Close Price'], color='b')
ax[1][1].plot(sim_5, color='r')
ax[1][2].plot(neww['Actual Nestle Close Price'], color='b')
ax[1][2].plot(sim_6, color='r')
ax[2][0].plot(neww['Actual Nestle Close Price'], color='b')
ax[2][0].plot(sim_7, color='r')
ax[2][1].plot(neww['Actual Nestle Close Price'], color='b')
ax[2][1].plot(sim_8, color='r')
ax[2][2].plot(neww['Actual Nestle Close Price'], color='b')
ax[2][2].plot(sim_9, color='r')
plt.show()#
print(neww['Actual Nestle Close Price'])