import matplotlib.pyplot as plt

from BMI_pyDeltaRCM import BmiDelta

delta = BmiDelta()
delta.initialize('input_configuration.yaml')

for _t in range(0, 50):
    delta.update()

delta.finalize()

fig, ax = plt.subplots()
ax.imshow(delta.get_value('sea_bottom_surface__elevation')
plt.show()
