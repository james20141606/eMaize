#! /bin/bash
PYRO_NS_PORT=9090
LogDir=Logs/lsi_model

detect_naming(){
    PYRO_NS_HOST=$(bjobs -w | grep 'lsi_model.naming' | awk '{print $6}')
    if [ -n "$PYRO_NS_HOST" ];then
        return 0
    else
        return 1
    fi
}

start_naming(){
    if detect_naming;then
        echo "Pyro naming server has been started at $PYRO_NS_HOST"
        exit 0
    fi
    [ -d "$LogDir" ] || mkdir -p $LogDir
    bsub <<BSUB
#BSUB -J lsi_model.naming
#BSUB -eo $LogDir/naming.stderr.log
#BSUB -oo $LogDir/naming.stdout.log
#BSUB -R "span[hosts=1]"
#BSUB -q Z-LU
#BSUB -n 1
export PYRO_SERIALIZERS_ACCEPTED=pickle
export PYRO_SERIALIZER=pickle
python -m Pyro4.naming -n 0.0.0.0 -p 9090
BSUB
}

start_workers(){
    local n_workers=${n_workers:=20}
    if ! detect_naming;then
        echo "Error: Pyro naming server has not been started"
        exit 1
    else
        echo "Start $n_workers workers using $PYRO_NS_HOST as the naming server"
    fi
    [ -d "$LogDir" ] || mkdir -p $LogDir
    bsub <<BSUB
#BSUB -J lsi_model.worker[1-$n_workers]
#BSUB -eo $LogDir/worker.%I.stderr.log
#BSUB -oo $LogDir/worker.%I.stdout.log
#BSUB -R "span[hosts=1] rusage[mem=10240]"
#BSUB -q Z-LU
#BSUB -n 1
export PYRO_NS_HOST=$PYRO_NS_HOST
export PYRO_NS_PORT=$PYRO_NS_PORT
export PYRO_SERIALIZERS_ACCEPTED=pickle
export PYRO_SERIALIZER=pickle
python -m gensim.models.lsi_worker
BSUB
}

start_dispatcher(){
    if ! detect_naming;then
        echo "Error: Pyro naming server has not been started"
        exit 1
    else
        echo "Start the dispatcher using $PYRO_NS_HOST as the naming server"
    fi
    [ -d "$LogDir" ] || mkdir -p $LogDir
    bsub <<BSUB
#BSUB -J lsi_model.dispatcher
#BSUB -eo $LogDir/dispatcher.stderr.log
#BSUB -oo $LogDir/dispatcher.stdout.log
#BSUB -R "span[hosts=1] rusage[mem=20480]"
#BSUB -q Z-LU
#BSUB -n 1
export PYRO_NS_HOST=$PYRO_NS_HOST
export PYRO_NS_PORT=$PYRO_NS_PORT
export PYRO_SERIALIZERS_ACCEPTED=pickle
export PYRO_SERIALIZER=pickle
python -m gensim.models.lsi_dispatcher
BSUB
}

start_job(){
    if [ -z "$1" ];then
        echo "Usage: $0 start_job script_file [args]"
        exit 1
    fi
    if ! detect_naming;then
        echo "Error: Pyro naming server has not been started"
        exit 1
    else
        echo "Start the job using $PYRO_NS_HOST as the naming server"
    fi
    export PYRO_NS_HOST
    export PYRO_NS_PORT=$PYRO_NS_PORT
    export PYRO_SERIALIZERS_ACCEPTED=pickle
    export PYRO_SERIALIZER=pickle
    python $@
}

submit_job(){
    : ${job_name:?}
    if [ -z "$1" ];then
        echo "Usage: $0 submit_job script_file [args]"
        exit 1
    fi
    if ! detect_naming;then
        echo "Error: Pyro naming server has not been started"
        exit 1
    else
        echo "Start the job using $PYRO_NS_HOST as the naming server"
    fi
    bsub <<BSUB
#BSUB -J $job_name
#BSUB -eo $LogDir/${job_name}.stderr.log
#BSUB -oo $LogDir/${job_name}.stdout.log
#BSUB -R "span[hosts=1] rusage[mem=40960]"
#BSUB -q Z-LU
#BSUB -n 1
export PYRO_NS_HOST=$PYRO_NS_HOST
export PYRO_NS_PORT=$PYRO_NS_PORT
export PYRO_SERIALIZERS_ACCEPTED=pickle
export PYRO_SERIALIZER=pickle
python $@
BSUB
}

stop_cluster(){
    echo "stop the dispatcher"
    bkill -J lsi_model.dispatcher
    echo "stop workers"
    bkill -J lsi_model.worker
    echo "stop the Pyro naming server"
    bkill -J lsi_model.naming
}

if [ -z "$1" ];then
    echo "Usage: $0 start_naming|start_workers|start_dispatcher|stop_cluster|start_job|submit_job"
    exit 1
fi

cmd=$1
shift
case "$cmd" in
    start_naming)
        start_naming
        ;;
    start_workers)
        start_workers
        ;;
    start_dispatcher)
        start_dispatcher
        ;;
    stop_cluster)
        stop_cluster
        ;;
    start_job)
        start_job $@
        ;;
    submit_job)
        submit_job $@
        ;;
esac
