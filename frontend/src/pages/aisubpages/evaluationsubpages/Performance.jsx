import React, { useState, useEffect } from "react"
import { useNavigate } from 'react-router-dom'
import axios from 'axios'
import ai from '../../assets/ai.svg'

const Performance = () => {

    const sideBar = {
        // "AI Performance": <Performance/>,
        // "Documentation": <Documentation/>,
        // "IPYNB Codebase": <Codebase/>,
        // "ML Models Used": <Models/>
        "AI Performance": 1,
        "Documentation": 1,
        "IPYNB Codebase": 1,
        "ML Models Used": 1,
    }

    const [active, setActive] = useState(Object.keys(sideBar)[0])

    return (
        <div className="flex flex-row w-full h-full bg-blue-500 gap-4">
            
        </div>
    )
}

export default Performance