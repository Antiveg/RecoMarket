const formatDate = (date) => {
    if(!date) return ''
    const d = new Date(date)
    return d.toLocaleString('en-US', {
        weekday: 'long',
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: 'numeric',
        minute: 'numeric',
        second: 'numeric',
        hour12: true
    })
}

module.exports = formatDate