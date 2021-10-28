"""In this script we store divergence fn for use."""
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfpl = tfp.layers
from hp import trainNo as trainingNumber
from hp import klMcSampleNo as klMcSampleNo 

divergence_fn_analytic =lambda q, p, ignore: kl_lib.kl_divergence(q, p)/trainingNumber

def kl_approx(q, p, _, n=klMcSampleNo):
    q_tensor = q.sample(n)
    return tf.reduce_mean(q.log_prob(q_tensor) - p.log_prob(q_tensor))

divergence_fn_approx = lambda q, p, _ : kl_approx(q, p, _) / trainingNumber


