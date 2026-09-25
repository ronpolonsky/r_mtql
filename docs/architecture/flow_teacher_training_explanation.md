# How the flow teacher is trained

The replay-buffer action chunk is the **data endpoint**, but it is not itself the teacher's regression label.

For each sampled replay tuple, the flow teacher uses only:

\[
(o, h, a)
\]

where:

- \(o\): current observation
- \(h\): observation history
- \(a\): stored action chunk, with shape `(chunk_length, action_dim)`

Rewards, masks, and next observations are not used by the flow loss.

## The three different action-like quantities

### 1. Data action chunk

\[
a \leftarrow \texttt{batch['actions']}
\]

This is the real action chunk stored in the replay buffer.

### 2. Interpolated or "noised" chunk

Random noise \(\epsilon\) is sampled with exactly the same shape as \(a\), along with a scalar time \(t\):

\[
\epsilon \sim \mathcal N(0,I)
\]

\[
x_t=(1-t)a+t\epsilon
\]

Therefore, \(x_t\) is a weighted interpolation—not simply \(a+\epsilon\).

- At \(t=0\): \(x_t=a\), the clean action chunk.
- At \(t=1\): \(x_t=\epsilon\), pure noise.
- Between them: \(x_t\) is partly action and partly noise.

### 3. Target velocity

The label predicted by the teacher is:

\[
u_t=\epsilon-a
\]

This vector points from the data action toward the sampled noise along the straight interpolation path.

## Teacher input, output, and loss

The teacher receives:

\[
(o,h,x_t,t)
\]

and predicts a velocity with the same shape as the action chunk:

\[
\hat v_t=v_{\theta_T}(o,h,x_t,t)
\]

It is trained with:

\[
\mathcal L_{\mathrm{flow}}
=\mathbb E\left[\left\|\hat v_t-(\epsilon-a)\right\|_2^2\right]
\]

So the roles are:

| Quantity | Meaning | Used as |
|---|---|---|
| \(a\) | Action chunk from replay | Clean endpoint |
| \(\epsilon\) | Random Gaussian chunk | Noise endpoint |
| \(x_t=(1-t)a+t\epsilon\) | Partially noised action chunk | Network input |
| \(u_t=\epsilon-a\) | Straight-path velocity | Regression label |
| \(\hat v_t\) | Teacher prediction | Compared with \(u_t\) |

## Small numerical example

For a one-dimensional action:

\[
a=2,\qquad \epsilon=-1,\qquad t=0.25
\]

the teacher input is:

\[
x_t=(1-0.25)(2)+0.25(-1)=1.25
\]

and the velocity label is:

\[
u_t=-1-2=-3
\]

The training example is therefore conceptually:

\[
(o,h,x_t=1.25,t=0.25)\longrightarrow u_t=-3
\]

## Why generation runs backward

The target velocity \(\epsilon-a\) points from actions at \(t=0\) toward noise at \(t=1\). At generation time, the model starts from noise and integrates with a negative time step from \(t=1\) to \(t=0\):

\[
x_{t-\Delta t}=x_t-\Delta t\,\hat v_t
\]

This reverses the learned direction and transforms noise into an action chunk.

## Code correspondence

The construction is implemented in [`actor_loss_flow_fn`](../../agents/mtql_transformer.py#L138-L159):

```python
actions = batch['actions']
noise = jax.random.normal(noise_rng, actions.shape)
x_t = time_expanded * noise + (1 - time_expanded) * actions
u_t = noise - actions
pred_vel = self.network.select('actor_bc_flow')(..., x_t, time[:, None], ...)
loss = jnp.mean((pred_vel - u_t) ** 2)
```

The key distinction is:

> The replay action \(a\) defines the clean endpoint; the supervised label is the constructed velocity \(\epsilon-a\).
