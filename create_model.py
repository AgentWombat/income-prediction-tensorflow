import tensorflow as tf
import data_parsing as dp
import json # This will allow us to save our weights


# Here we implement Logistic regression with mutliple input variables--
# (age, has_degree, is_married, sex, hours_of_work_week)


def call(x, weights: tf.Tensor):
    '''
    Calls the model
    :param x: The inputs
    :param weights: A list with length 6 of weights (numbers)
    :returns: The outputs
    '''
    # Linear combination of features
    s = (x[:, 0] * weights[0] + x[:,1]*weights[1] + x[:,2]*weights[2] + x[:,3]*weights[3] +
        x[:,4]*weights[4] + weights[5])

    # The sigmoid function "squishes" all of our outputs to be between 0 and 1.
    a = tf.sigmoid(s)

    return a

def binary_crossentropy(y, y_hat):
    
    # This clever construction is bassically a differentiable "if-else" statement.
    # It depends on the fact that values in 'y' will always be either 0 or 1.
    # If the actual value in 'y' is 0, it cancels the first term through multiplication by 0.
    # If the actual value in 'y' is 1, (1-y) evaluates to 0 for that entree,
    # cancelling the entire second term.
    # So, in the event that y = 0, we have a term which punishes the model for having a value
    # close to 1; in the event that y = 1, we have a term which punishes it for having a value
    # close to 0. 
    loss = -( y * tf.math.log(y_hat) + (1 - y) * tf.math.log(1 - y_hat) )
    
    # Get average (sum all values in 'loss' and divide by number of values)
    loss = tf.reduce_mean(loss)
    return loss

# We need to ensure that all of our features have about the same range of values.
# When we do this, neural networks train MUCH faster in many cases.
def standardize(x):
    '''
    Standardizes data. That is, this function makes most features be between 0 and 1.
    :returns: The standardized data.
    '''

    x_st = tf.Variable(x)

    x_st[:, 0].assign(x[:,0] / dp.MAX_AGE)

    x_st[:, 4].assign(x[:,4] / dp.MAX_HOURS)

    return x_st

# Get data and standardize it
x_train, y_train, x_test, y_test = dp.load_data()
x_train = standardize(x_train)
x_test = standardize(x_test)

# TRAINING
# This creates a random assortment of gradients (weights)
weights = tf.Variable(tf.random.uniform(shape = (6,)))

for i in range(200):

    with tf.GradientTape() as tape:

        # Forward propagate
        y_hat = call(x_train, weights)
        loss = binary_crossentropy(y_train, y_hat)

    # Get gradients (derivatives)
    grads = tape.gradient(loss, weights)

    # Update weights
    lr = 1
    weights.assign(weights - grads * lr)

    # Display training progress every 25 epochs
    if i % 25 == 0:
        print("EPOCH", i + 1, "LOSS:",loss.numpy())



# Print model weights and test model accuracy on never-before-seen data.
print("weights:",weights.numpy())

y_hat = call(x_test, weights)
y_hat = tf.math.round(y_hat)
num_correct = tf.math.reduce_sum(1 - tf.math.abs(y_test - y_hat))

print("TESTING ACCURACY: ", num_correct / len(y_test))


# Save weights to file
with open('weights.json', 'w') as file:
    json.dump([float(num) for num in weights.numpy()], file)

print("Weights saved to \"weights.json\"")








