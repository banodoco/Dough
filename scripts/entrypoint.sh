#!/bin/bash

COMMAND="streamlit run app.py --runner.fastReruns false --server.runOnSave true --server.port 5500"

compare_versions() {
    IFS='.' read -r -a ver1 << EOF
$(echo "$1")
EOF
    IFS='.' read -r -a ver2 << EOF
$(echo "$2")
EOF

    # Fill empty fields with zeros
    i=${#ver1[@]}
    while [ $i -lt ${#ver2[@]} ]; do
        ver1[i]=0
        i=$((i + 1))
    done
    i=0
    while [ $i -lt ${#ver2[@]} ]; do
        if [ -z "${ver1[i]}" ]; then
            ver1[i]=0
        fi
        i=$((i + 1))
    done

    # Compare major and minor versions
    i=0
    while [ $i -lt $((${#ver1[@]} - 1)) ]; do
        if [ $((10#${ver1[i]})) -gt $((10#${ver2[i]})) ]; then
            echo "1"
            return
        elif [ $((10#${ver1[i]})) -lt $((10#${ver2[i]})) ]; then
            echo "-1"
            return
        fi
        i=$((i + 1))
    done

    # Compare patch versions
    if [ $((10#${ver1[2]})) -gt $((10#${ver2[2]})) ]; then
        echo "1"
        return
    elif [ $((10#${ver1[2]})) -lt $((10#${ver2[2]})) ]; then
        echo "-1"
        return
    fi

    echo "0"
}

update_app() {
    CURRENT_VERSION=$(curl -s "https://raw.githubusercontent.com/banodoco/Dough/feature/final/scripts/app_version.txt")

    ERR_MSG="Unable to fetch the current version from the remote repository."
    # file not present
    if ! echo "$CURRENT_VERSION" | grep -q '^[0-9]\+\.[0-9]\+\.[0-9]\+$'; then
        echo $ERR_MSG
        return
    fi
    # file is empty
    if [ -z "$CURRENT_VERSION" ]; then
        echo $ERR_MSG
        return
    fi

    CURRENT_DIR=$(pwd)
    LOCAL_VERSION=$(cat ${CURRENT_DIR}/scripts/app_version.txt)
    echo "local version $LOCAL_VERSION"
    echo "current version $CURRENT_VERSION"
    VERSION_DIFF=$(compare_versions "$LOCAL_VERSION" "$CURRENT_VERSION")
    if [ "$VERSION_DIFF" == "-1" ]; then
        echo "A newer version ($CURRENT_VERSION) is available. Updating..."

        git stash
        # Step 1: Pull from the current branch
        git pull origin "$(git rev-parse --abbrev-ref HEAD)"

        # Step 2: Check if the comfy_runner folder is present
        if [ -d "${CURRENT_DIR}/comfy_runner" ]; then
            # Step 3a: If comfy_runner is present, pull from the feature/package branch
            # echo "comfy_runner folder found. Pulling from feature/package branch."
            cd comfy_runner && git pull origin main
            cd ..
        else
            # Step 3b: If comfy_runner is not present, clone the repository
            echo "comfy_runner folder not found. Cloning repository."
            REPO_URL="https://github.com/piyushK52/comfy_runner.git"
            git clone "$REPO_URL" "${CURRENT_DIR}/comfy_runner"
        fi

        echo "$CURRENT_VERSION" > ${CURRENT_DIR}/scripts/app_version.txt
    else
        echo "You have the latest version ($LOCAL_VERSION)."
    fi
}

while [ "$#" -gt 0 ]; do
    case $1 in
        --update)
            update_app
            ;;
        *)
            echo "Invalid option: $1" >&2
            exit 1
            ;;
    esac
    shift
done

# Execute the base command
eval $COMMAND