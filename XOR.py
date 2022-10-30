import matplotlib.pyplot as plt

from AI.NN.NN_Network2 import Net2D

from AI.ML.ML_Function import *

from AI.Rest.Help import *
from AI.Data.Data_XOR import *
from AI.Data.Data_Points import *


# ================================================================================


X, Y = xor_data()
print(f"X: {X.shape}")
print(f"Y: {Y.shape}")

# ================================================================================

# np.random.seed(1)

# bone[0] --> input
bone = [2,
        4, 1]

func = [tan_h, sig]

# ================================================================================

model = Net2D(bone)
model.F = func

model.init_parameters()
model.X = X
model.Y = Y

test_init_loss(model, 3)
test_init_optimizer(model, 4)

model.init_optimizers()

# ================================================================================

acc = 10000
model.train(acc)
history = model.history

# ================================================================================

print()
print("[0, 0] -> ?: " + str(model.predict(np.array([0, 0]))))
print("[0, 1] -> ?: " + str(model.predict(np.array([0, 1]))))
print("[1, 1] -> ?: " + str(model.predict(np.array([1, 1]))))
print("[1, 0] -> ?: " + str(model.predict(np.array([1, 0]))))

# ================================================================================

s = 0.1
plt.ylim(0 - s, 1 + s)
plt.plot(history, '-b')

plt.plot([0, len(history) - 1], [0, 0], '-k')

plt.get_current_fig_manager().toolbar.pan()
plt.grid(True)
plt.show()
