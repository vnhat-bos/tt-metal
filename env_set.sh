export ARCH_NAME=blackhole
export TT_METAL_DISABLE_L1_DATA_CACHE_RISCVS="BR,NC,TR,ER"
export TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE="7,5" # for 8x6 core grid
export TT_METAL_ENV=dev
export TT_METAL_HOME=$(pwd)
export WORKING_DIR=$TT_METAL_HOME/models/bos_model/ssr
export BOS_METAL_HOME=$TT_METAL_HOME/tt_metal/third_party/bos-metal
export PYTHONPATH=$TT_METAL_HOME:$BOS_METAL_HOME:$PYTHONPATH:$WORKING_DIR:SSR

# Parse command line arguments to determine Python environment directory
if [ $# -gt 1 ]; then
    echo "Error: Too many arguments. Usage: source $0 [ssr]"
    return 1
fi

if [ $# -eq 1 ]; then
    if [ "$1" = "ssr" ]; then
        export PYTHON_ENV_DIR=$TT_METAL_HOME/python_env_ssr
    else
        echo "Error: Invalid argument '$1'. Only 'ssr' is allowed."
        return 1
    fi
else
    export PYTHON_ENV_DIR=$TT_METAL_HOME/python_env
fi
# export TT_METAL_LOGGER_LEVEL="Debug"
# export TT_METAL_LOGGER_TYPES="All"
# export TT_METAL_DPRINT_CHIPS=0
# export TT_METAL_DPRINT_CORES=0,0
# export TT_METAL_KERNEL_READBACK_ENABLE="1"

if [ -f "$PYTHON_ENV_DIR/bin/activate" ]; then
    source $PYTHON_ENV_DIR/bin/activate
fi
