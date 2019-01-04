import json
li = ['A chance of rain showers before 10pm . Snow level 1000 feet . Mostly cloudy , with a low around 32 . Southwest wind between 5 and 7 mph . Chance of precipitation is 30 % . \n',
'Mostly cloudy , with a low around 41 . Southwest wind around 5 mph becoming calm . \n','A chance of drizzle before 10am . Mostly cloudy , with a high near 70 . South wind between 7 and 14 mph . \n']
print(li)
d  = {}
d_li = []
with open('j_out', 'w') as fp:
	for i in range(len(li)):
		d['image_id'] = i
		d['caption'] = li[i]
		d_li.append(d)

	json.dump(d_li,fp,indent=4)	
