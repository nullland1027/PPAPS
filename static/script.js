function displayFileName() {
    const fileInput = document.getElementById('file');
    const fileName = fileInput.files[0].name;
    const fileNameDisplay = document.getElementById('fileName');
    fileNameDisplay.innerHTML = `File selected：${fileName}`;
}

function checkFileSelected() {
    const fileInput = document.querySelector('input[type=file]');
    if (!fileInput.value) {
        alert('Please select a file to upload.');
        return false;
    }
    return true;
}

function validateForm() {
    let i;
    const radio_kind = document.getElementsByName("kind");
    const radio_al = document.getElementsByName("al");
    let formValid_kind = false;
    let formValid_al = false;
    for (i = 0; i < radio_kind.length; i++) {
        if (radio_kind[i].checked) {
            formValid_kind = true;
            break;
        }
    }
    for (i = 0; i < radio_al.length; i++) {
        if (radio_al[i].checked) {
            formValid_al = true;
            break;
        }
    }

    if (!formValid_kind || !formValid_al) {
        alert("Please select an option!");
        return false;
    }
    return true;
}


function validatePassword() {
    var password = document.getElementsByName("password")[0].value;
    var confirmPassword = document.getElementsByName("confirm-password")[0].value;

    if (password !== confirmPassword) {
        document.getElementsByName("password")[0].classList.add("border-danger");
        document.getElementsByName("password")[0].style.borderWidth = "4px";
        document.getElementsByName("confirm-password")[0].classList.add("border-danger");
        document.getElementsByName("confirm-password")[0].style.borderWidth = "4px";
        alert('Password Not Match!')
        return false;
    } else {
        document.getElementsByName("password")[0].classList.remove("border-danger");
        document.getElementsByName("confirm-password")[0].classList.remove("border-danger");
        return true;
    }
}

function signUp() {
    if (!validatePassword()) {
        return
    }
    $.ajax({
        type: "POST",
        url: "/afterSignUp",
        data: {
            email: $("#email").val(),
            password: $("#floatingPassword").val()
        },
        success: function (response) {
            alert(response['message']);
            window.location.assign("/login");
        },
        error: function (xhr, status, error) {
            document.getElementById('email').style.borderWidth = "4px";
            document.getElementById('email').classList.add("border-danger");
            if (xhr.status === 400) {
                alert(xhr.responseJSON['message']);
            } else {
                console.log("Error:", error);
            }
        }
    })
}

