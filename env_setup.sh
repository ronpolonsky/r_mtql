  export UV_CACHE_DIR=/iris/u/marcelto/.cache/uv
  export HOME=/iris/u/marcelto
  # MuJoCo
  export MUJOCO_PATH=/iris/u/marcelto/.mujoco
  export MUJOCO_PY_MUJOCO_PATH=/iris/u/marcelto/.mujoco/mujoco210
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/iris/u/marcelto/.mujoco/mujoco210/bin                                                                 
   
  # NVIDIA                                                                                                                                  
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia   
                                                                                                                                              
  # CUDA / JAX                                                                                                                                
  export SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")                                                           
  export PATH="$SITE_PACKAGES/nvidia/cuda_nvcc/bin:$PATH"                                                                                     
  export LD_LIBRARY_PATH="$SITE_PACKAGES/nvidia/cuda_runtime/lib:$SITE_PACKAGES/nvidia/cudnn/lib:$LD_LIBRARY_PATH"                            
  export XLA_PYTHON_CLIENT_PREALLOCATE=false                                                                                                  
  export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda                                                                                    
                                                                                                                                              
  # MuJoCo rendering (headless server)                                                                                                        
  export MUJOCO_GL=egl                                                                                                                        
  export DISPLAY=                                                                                                                             
  export CFLAGS="-DGLEW_NO_GLU ${CFLAGS:-}"
  export CXXFLAGS="-DGLEW_NO_GLU ${CXXFLAGS:-}"                                                                                               
