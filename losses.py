import torch
import torch.nn as nn
import torch.nn.functional as F


def log_t(u, t):
    """Compute log_t for `u`."""
    if t == 1.0:
        return torch.log(u)
    else:
        return (u ** (1.0 - t) - 1.0) / (1.0 - t)


def exp_t(u, t):
    """Compute exp_t for `u`."""
    if t == 1.0:
        return torch.exp(u)
    else:
        return torch.relu(1.0 + (1.0 - t) * u) ** (1.0 / (1.0 - t))


def compute_normalization_fixed_point(activations, t, num_iters=5):
    """Returns the normalization value for each example (t > 1.0).
    Args:
      activations: A multi-dimensional tensor with last dimension `num_classes`.
      t: Temperature 2 (> 1.0 for tail heaviness).
      num_iters: Number of iterations to run the method.
    Return: A tensor of same shape as activation with the last dimension being 1.
    """
    mu = torch.max(activations, dim=-1).values.view(-1, 1)
    normalized_activations_step_0 = activations - mu

    normalized_activations = normalized_activations_step_0
    for _ in range(num_iters):
        logt_partition = torch.sum(exp_t(normalized_activations, t), dim=-1).view(-1, 1)
        normalized_activations = normalized_activations_step_0 * (logt_partition ** (1.0 - t))

    logt_partition = torch.sum(exp_t(normalized_activations, t), dim=-1).view(-1, 1)
    return -log_t(1.0 / logt_partition, t) + mu


def compute_normalization(activations, t, num_iters=5):
    """Returns the normalization value for each example."""
    if t < 1.0:
        return None  # not implemented as these values don't occur in the paper
    else:
        return compute_normalization_fixed_point(activations, t, num_iters)


def tempered_softmax(activations, t, num_iters=5):
    """Tempered softmax function.
    Args:
      activations: A multi-dimensional tensor with last dimension `num_classes`.
      t: Temperature > 1.0.
      num_iters: Number of iterations to run the method.
    Returns:
      A tensor of same shape as activation with the last dimension being 1.
    """
    if t == 1.0:
        return F.softmax(activations, dim=-1)

    normalization_constants = compute_normalization(activations, t, num_iters)
    return exp_t(activations - normalization_constants, t)


def bi_tempered_logistic_loss(activations, labels, t1, t2, num_iters=5, label_smoothing=0.0):
    """Bi-Tempered Logistic Loss with custom gradient.
    Args:
      activations: A multi-dimensional tensor with last dimension `num_classes`.
      labels: A tensor with shape and dtype as activations.
      t1: Temperature 1 (< 1.0 for boundedness).
      t2: Temperature 2 (> 1.0 for tail heaviness).
      num_iters: Number of iterations to run the method.
    Returns:
      A loss tensor.
    """
    if t2 == 1.0:
        loss_values = F.cross_entropy(activations, labels)
        return loss_values

    # Convert labels to one-hot
    num_classes = activations.size(-1)
    labels_onehot = F.one_hot(labels, num_classes).float()

    if label_smoothing > 0.0:
        labels_onehot = (1 - label_smoothing) * labels_onehot + label_smoothing / num_classes

    probabilities = tempered_softmax(activations, t2, num_iters)

    temp1 = (labels_onehot * (1.0 - probabilities ** (1.0 - t1))) / (1.0 - t1)
    temp2 = ((1.0 - labels_onehot) * (1.0 - (1.0 - probabilities) ** (1.0 - t1))) / (1.0 - t1)
    loss_values = temp1 + temp2

    return loss_values.sum(dim=-1).mean() 