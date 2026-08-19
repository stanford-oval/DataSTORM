import docker
import io
import base64
import pickle

# Initialize Docker client (connects to local Docker daemon)
client = docker.from_env()

# Your sandbox container name
CONTAINER_NAME = "llm-sandbox"

def execute_python_from_sql_results(sql_results, python_code):
    
    # print(f"execute_python_from_sql_results: sql_results: {sql_results}")
    # print(f"execute_python_from_sql_results: python_code: {python_code}")
    
    # Serialize sql_results using pickle (handles non-JSON-serializable Python objects)
    pickled_bytes = pickle.dumps(sql_results, protocol=pickle.HIGHEST_PROTOCOL)
    pickled_b64 = base64.b64encode(pickled_bytes).decode("ascii")

    # Inject bootstrap and deserialize inside the sandbox
    # Also install a Matplotlib hook so user code can simply call plt.show(),
    # and we will emit base64-encoded PNGs to stdout with clear sentinels.
    injected_code = (
        "import base64, pickle\n"
        "import pandas as pd\n"
        "import io, sys\n"
        "import warnings\n"
        "warnings.filterwarnings('ignore', '.*numpy.core.numeric.*', DeprecationWarning)\n"
        "try:\n"
        "    import matplotlib\n"
        "    matplotlib.use('Agg')\n"
        "    import matplotlib.pyplot as plt\n"
        "    def __datatalk_emit_plot(fig):\n"
        "        buf = io.BytesIO()\n"
        "        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')\n"
        "        buf.seek(0)\n"
        "        data = base64.b64encode(buf.getvalue()).decode('ascii')\n"
        "        sys.stdout.write('<<KRAKEN_PLOT:PNG>>' + data + '<<END>>\\n')\n"
        "        sys.stdout.flush()\n"
        "    def __datatalk_show(*args, **kwargs):\n"
        "        try:\n"
        "            import matplotlib._pylab_helpers as pylab_helpers\n"
        "            managers = pylab_helpers.Gcf.get_all_fig_managers()\n"
        "            if not managers:\n"
        "                return\n"
        "            for m in managers:\n"
        "                __datatalk_emit_plot(m.canvas.figure)\n"
        "        except Exception as e:\n"
        "            sys.stdout.write(f'<<KRAKEN_PLOT_ERROR>>{e}<<END>>\\n')\n"
        "            sys.stdout.flush()\n"
        "    # Monkey-patch show so any plt.show() in user code triggers our emitter\n"
        "    plt.show = __datatalk_show\n"
        "except Exception:\n"
        "    # If matplotlib is unavailable, proceed without hooking\n"
        "    pass\n"
        f"sql_results = pickle.loads(base64.b64decode(\"{pickled_b64}\"))\n"
        f"{python_code}"
    )

    exec_result = client.containers.get(CONTAINER_NAME).exec_run(
        cmd=["python", "-c", injected_code],
        user="sandbox",  # enforce non-root execution
        stdout=True,
        stderr=True
    )
    return exec_result.output.decode()


if __name__ == "__main__":
    # Minimal test driving the sandbox runner (pass a DataFrame as sql_results)
    import pandas as pd
    sample_df = pd.DataFrame([
        {"month": "2024-01-01", "incident_count": 3},
        {"month": "2024-02-01", "incident_count": 5},
        {"month": "2024-03-01", "incident_count": 2},
    ])

    good_code = (
        "print('rows=', len(sql_results))\n"
        "print(sql_results.to_string(index=False))\n"
    )

    print("=== RUNNING GOOD CODE (DataFrame input) ===")
    print(execute_python_from_sql_results(sample_df, good_code))

    bad_code = (
        "print('this will cause a syntax error'\n"  # missing closing parenthesis
    )

    print("=== RUNNING SYNTAX ERROR CODE (DataFrame input) ===")
    print(execute_python_from_sql_results(sample_df, bad_code))

    # Plotting test: ensure plt.show() is captured and returns a PNG sentinel
    plot_code = (
        "import matplotlib.pyplot as plt\n"
        "import pandas as pd\n"
        "sql_results['month'] = pd.to_datetime(sql_results['month'])\n"
        "plt.figure(figsize=(4,3))\n"
        "plt.plot(sql_results['month'], sql_results['incident_count'], marker='o')\n"
        "plt.title('Trend of Incidents Over Time')\n"
        "plt.tight_layout()\n"
        "plt.show()\n"
    )

    print("=== RUNNING PLOTTING CODE (DataFrame input) ===")
    plot_output = execute_python_from_sql_results(sample_df, plot_code)
    print(plot_output)
    if '<<KRAKEN_PLOT:PNG>>' in plot_output:
        print('PLOT_CAPTURED=1')