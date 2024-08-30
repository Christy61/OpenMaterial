function showContent(section) {
    const sections = ['Home', 'Preview', 'Download', 'Benchmark-Board', 'Visualization'];
    sections.forEach(sec => {
        document.getElementById(`content-${sec}`).classList.remove('visible');
        document.querySelector(`.button-card[onclick="showContent('${sec}')"]`).classList.remove('active');
    });
    document.getElementById(`content-${section}`).classList.add('visible');
    document.querySelector(`.button-card[onclick="showContent('${section}')"]`).classList.add('active');
}

document.addEventListener("DOMContentLoaded", () => {
    showContent('Home');
});