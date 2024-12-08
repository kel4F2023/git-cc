import os
import sys
import argparse
import subprocess
import signal
import shutil
from pathlib import Path
import joblib
from colorama import Fore, Style, Back
from InquirerPy import inquirer
from .settings import Settings

LOG_HEADING = f"[GIT-CC]"


def format_info(message):
    return f"{Fore.CYAN}{LOG_HEADING}{Style.RESET_ALL}: {message}"


def format_error(message):
    return f"{Fore.RED}{LOG_HEADING}{Style.RESET_ALL}: {message}"


def format_success(message):
    return f"{Fore.GREEN}{LOG_HEADING}{Style.RESET_ALL}: {message}"


def format_warning(message):
    return f"{Fore.YELLOW}{LOG_HEADING}{Style.RESET_ALL}: {message}"


def signal_handler(sig, frame):
    print(f"\n" + format_error(f"{Fore.RED}Commit process terminated.{Style.RESET_ALL}"))
    exit(0)


def get_available_models():
    """Get list of available model files in the model directory"""
    model_dir = Path(__file__).parent / "model"
    if not model_dir.exists():
        return []
    # Remove .joblib extension and map to display names
    models = [f.stem for f in model_dir.glob("*.joblib")]
    return sorted(models)


def get_model_path(model_name=None):
    """Get the path to the model file from the installed package"""
    try:
        model_dir = Path(__file__).parent / "model"
        
        # If specific model is requested, return its path if it exists
        if model_name:
            model_path = model_dir / f"{model_name}.joblib"
            if model_path.exists():
                return str(model_path)
            raise FileNotFoundError(f"Model {model_name} not found")

        # Default model selection logic - use mini as default
        if (model_dir / "mini.joblib").exists():
            return str(model_dir / "mini.joblib")
        elif (model_dir / "advanced.joblib").exists():
            return str(model_dir / "advanced.joblib")
        else:
            raise FileNotFoundError("No model files found. Please ensure at least one model is installed.")
    except Exception as e:
        print(format_error(f"Error locating model file: {e}"), file=sys.stderr)
        sys.exit(1)


def git_commit(message, commit_type):
    """Create a git commit with the given type and message"""
    formatted_message = f"{commit_type}: {message}"
    try:
        subprocess.run(['git', 'commit', '-m', formatted_message], check=True)

    except subprocess.CalledProcessError as e:
        print(f"Error creating commit: {e}", file=sys.stderr)
        sys.exit(1)


def copy_model_file(source_path):
    """Copy the model file from the specified source to the default package location."""
    try:
        # Resolve the source path
        source = Path(source_path).resolve()
        if not source.exists():
            print(f"Error: The specified model file does not exist: {source}", file=sys.stderr)
            sys.exit(1)

        # Ensure the target directory exists
        if os.path.exists(Path(__file__).parent):
            shutil.copy(source, Path(__file__).parent / "model" / "custom_classifier.joblib")
            print(format_success(f"Model successfully loaded"))

    except Exception as e:
        print(format_error(f"Error while copying model file: {e}", file=sys.stderr))
        sys.exit(1)


class CommitClassifier:
    def __init__(self, model_name=None):
        self.pipeline = None
        self.load_model(model_name)

    def load_model(self, model_name=None):
        """Load the pre-trained model"""
        try:
            model_path = get_model_path(model_name)
            loaded_data = joblib.load(model_path)
            
            # Handle both dictionary and direct pipeline formats
            if isinstance(loaded_data, dict):
                self.pipeline = loaded_data.get('model')
                if self.pipeline is None:
                    raise ValueError("Invalid model format: 'model' key not found")
            else:
                self.pipeline = loaded_data
                
        except Exception as e:
            print(format_error(f"Error loading model: {e}"), file=sys.stderr)
            sys.exit(1)

    def predict(self, message):
        """Predict commit type for a message"""
        try:
            if self.pipeline is None:
                raise ValueError("Model not properly loaded")
            return self.pipeline.predict([message])[0]
        except Exception as e:
            print(format_error(f"Error predicting commit type: {e}"), file=sys.stderr)
            return 'chore'  # Default fallback


def list_models(settings):
    """List available models with currently selected model marked"""
    models = get_available_models()
    selected_model = settings.get_selected_model()
    
    if models:
        print(format_info("Available models:"))
        for model in models:
            description = ""
            if model == "mini":
                description = f"{Style.DIM}(Lightweight TF-IDF with Naive Bayes classifier){Style.RESET_ALL}"
            elif model == "advanced":
                description = f"{Style.DIM}(Advanced Random Forest with metadata features){Style.RESET_ALL}"
            
            if model == selected_model:
                print(f"  {Fore.GREEN}✓{Style.RESET_ALL} {Fore.CYAN}{model}{Style.RESET_ALL} {description}")
            else:
                print(f"    {Fore.CYAN}{model}{Style.RESET_ALL} {description}")
    else:
        print(format_warning("No models available"))


def select_model(settings):
    """Interactively select a model and save the selection"""
    models = get_available_models()
    if not models:
        print(format_error("No models available to select from"))
        sys.exit(1)
    
    current_model = settings.get_selected_model()
    choices = models.copy()
    
    # Format choices to show current selection with descriptions
    choices_display = []
    for model in models:
        display = model
        if model == "mini":
            display = f"{model} - Lightweight TF-IDF with Naive Bayes classifier"
        elif model == "advanced":
            display = f"{model} - Advanced Random Forest with metadata features"
            
        if model == current_model:
            display = f"{display} {Fore.GREEN}(current){Style.RESET_ALL}"
        choices_display.append(display)
    
    selected = inquirer.select(
        message="Select the model to use:",
        choices=choices_display,
        default=models.index(current_model) if current_model in models else 0,
    ).execute()
    
    # Extract the model name from the selection (remove description and current marker)
    selected_model = selected.split(' - ')[0]
    settings.set_selected_model(selected_model)
    print(format_success(f"Selected model: {Fore.CYAN}{selected_model}{Style.RESET_ALL}"))
    return selected_model


def main():
    signal.signal(signal.SIGINT, signal_handler)
    settings = Settings()

    parser = argparse.ArgumentParser(
        description='Git Conventional Commit Classifier',
        usage='git cc -m <message> [--load <model_path>] [--list-models] [--select-model]'
    )

    parser.add_argument('--message', '-m', help='Commit message')
    parser.add_argument("--load", help="Path to load a custom model", default=None)
    parser.add_argument("--list-models", action="store_true", help="List available models")
    parser.add_argument("--select-model", action="store_true", help="Interactively select a model to use")
    args = parser.parse_args()

    # Handle utility commands first
    if args.list_models:
        list_models(settings)
        sys.exit(0)

    if args.load:
        copy_model_file(args.load)
        sys.exit(0)

    if args.select_model:
        select_model(settings)
        print(format_success("Model selection saved successfully!"))
        sys.exit(0)

    # Handle commit operation
    if not args.message:
        print(format_error("Please provide a commit message using -m option"))
        sys.exit(1)

    selected_model = settings.get_selected_model()
    classifier = CommitClassifier(selected_model)
    commit_type = classifier.predict(args.message)

    try:
        user_input = input(
            format_info(
                f"Do you want to proceed with commit type '{Back.GREEN}{Style.BRIGHT}{commit_type}{Style.RESET_ALL}'? [Y/n]: ")
        ).strip().lower()

        if user_input in ('yes', 'y', ''):  # Accepts Yes, Y, or Enter as confirmation
            git_commit(args.message, commit_type)
            print(format_success(f"{Fore.GREEN}Commit process completed.{Style.RESET_ALL}"))
        else:
            print(format_info(f"{Fore.YELLOW}You chose not to proceed with the predicted type.{Style.RESET_ALL}"))

            # Commit types for selection
            commit_types = ['feat', 'fix', 'chore', 'docs', 'style', 'refactor', 'test', 'perf', 'ci']

            # Interactive selection using InquirerPy
            commit_type = inquirer.select(
                message="Select the commit type:",
                choices=commit_types,
                default=commit_type,
            ).execute()

            print(format_info(f"Selected commit type: {Fore.CYAN}{commit_type}{Style.RESET_ALL}"))

            # Ask for final confirmation after manual selection
            user_input = input(
                format_info(f"Do you want to proceed with commit type '{Fore.GREEN}{commit_type}{Style.RESET_ALL}'? [Yes/No]: ")).strip().lower()
            if user_input in ('yes', 'y', ''):
                git_commit(args.message, commit_type)
                print(format_success(f"{Fore.GREEN}Commit process completed.{Style.RESET_ALL}"))
            else:
                print(format_error(f"{Fore.RED}Commit process terminated.{Style.RESET_ALL}"))

    except KeyboardInterrupt:
        sys.exit(0)

    finally:
        sys.exit(0)


if __name__ == '__main__':
    main()
