function showContent(section) {
    const sections = ['Preview', 'Statistics-and-Distribution', 'Novel-View-Synthesis', '3D-Shape-Reconstruction', 'References'];
    sections.forEach(sec => {
        document.getElementById(`content-${sec}`).classList.remove('visible');
        document.querySelector(`.button-card[onclick="showContent('${sec}')"]`).classList.remove('active');
    });
    document.getElementById(`content-${section}`).classList.add('visible');
    document.querySelector(`.button-card[onclick="showContent('${section}')"]`).classList.add('active');
}

document.addEventListener("DOMContentLoaded", () => {
    showContent('Preview');
});