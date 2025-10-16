import torch
import numpy as np




def unwrap_env(env):
    """Récupère l'environnement sous-jacent contenant .sim"""
    while hasattr(env, "env"):
        env = env.env
    return env

def get_box2d_state(env):
    state = {
        "lander": (
            env.lander.position.copy(),
            env.lander.linearVelocity.copy(),
            env.lander.angle,
            env.lander.angularVelocity,
        ),
        "legs": [
            (leg.ground_contact, leg.position.copy(), leg.linearVelocity.copy())
            for leg in env.legs
        ],
    }
    return state

def set_box2d_state(env, state):
    env.lander.position = state["lander"][0].copy()
    env.lander.linearVelocity = state["lander"][1].copy()
    env.lander.angle = state["lander"][2]
    env.lander.angularVelocity = state["lander"][3]

    for leg, (contact, pos, vel) in zip(env.legs, state["legs"]):
        leg.ground_contact = contact
        leg.position = pos.copy()
        leg.linearVelocity = vel.copy()


def Monte_Carlo(s, a, envs, actor, args, max_step, N):


    base_env = unwrap_env(envs.envs[0])
    state_data = get_box2d_state(base_env)

    etas_init = s.copy()
    actions_init = a.copy()
    list_G0 = []
    
    for i in range(N):
        set_box2d_state(base_env, state_data)

        next_etas_init, reward, terminated, truncated, info = base_env.step(actions_init)
        done = terminated or truncated

        discount_factor = 1.0

        G0 = reward
        etas_init = next_etas_init

        j = 0
        
        while not done and j < max_step:
            etas_tensor = torch.tensor(etas_init, dtype=torch.float32).unsqueeze(0)
            
            etas_tensor = etas_tensor.to("cuda")
            with torch.no_grad():
                action = actor(etas_tensor).cpu().numpy()[0]  
            etas_init, reward, terminated, truncated, info = base_env.step(action)
            done = terminated or truncated
            
            G0 += discount_factor*reward 
            discount_factor *= args.gamma
            j +=1 
        etas_init = s.copy()
        actions_init = a.copy()
        list_G0.append(G0)

    G_pi = np.mean(list_G0)

    set_box2d_state(base_env, state_data)

    return G_pi, list_G0